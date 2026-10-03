// NVDEC (H.264) and VIC engine execution.
//
// NVDEC: the guest's multimedia library parses the stream itself and hands the engine a picture
// info block (H264 parameter set + DPB) plus the slice data. We rebuild SPS/PPS from the parameter
// set, prepend them to the slice data and decode with the Windows Media Foundation H.264 decoder.
// Decoded frames are kept host-side keyed by the output surface's luma address; VIC looks them up
// by its input surface address, converts YUV -> RGB and writes the guest output surface.
#include <windows.h>
#include <codecapi.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>
#include <wmcodecdsp.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <set>
#include <thread>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "gpu/gpu.h"
#include "host1x.h"

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "wmcodecdspuuid.lib")
#pragma comment(lib, "ole32.lib")

namespace video {

namespace {

// ------------------------------------------------------------------ guest memory via IOVA
bool read_iova(u32 iova, void* dst, u64 size) {
    u64 a = 0, avail = 0;
    if (!iova_to_guest(iova, &a, &avail) || avail < size) return false;
    memcpy(dst, (const void*)a, size);
    return true;
}
u8* iova_ptr(u32 iova, u64 size) {
    u64 a = 0, avail = 0;
    if (!iova_to_guest(iova, &a, &avail) || avail < size) return nullptr;
    return (u8*)a;
}

// ------------------------------------------------------------------ decoded frames
struct Frame {
    u32 width = 0, height = 0;  // coded size
    std::vector<u8> y, uv;      // NV12, stride = width
    u32 session = 0;            // decode session (movie) this frame belongs to
};
// A "session" is one continuous stream: a gap of 400+ ms between decode jobs starts a new one, so a
// frame of the previous movie is never shown in place of a not-yet-decoded frame of the next.
static std::atomic<u32> g_session{0};
static u64 g_last_job_ms = 0;
static u32 g_stat_hits = 0, g_stat_waits = 0, g_stat_fallback = 0, g_stat_none = 0;
static double g_stat_max_wait = 0;
std::mutex g_frames_lock;
std::condition_variable g_frames_cv;
std::map<u32, std::shared_ptr<Frame>> g_frames;  // by output luma address
std::shared_ptr<Frame> g_last_frame;
std::multiset<u32> g_pending;  // luma tags whose decode job has not finished yet

void put_frame(u32 luma, std::shared_ptr<Frame> f) {
    {
        std::lock_guard<std::mutex> l(g_frames_lock);
        f->session = g_session.load();
        g_frames[luma] = f;
        g_last_frame = f;
        while (g_frames.size() > 64) g_frames.erase(g_frames.begin());
    }
    g_frames_cv.notify_all();
}
// The frame decoded into `luma`; waits (bounded) for an in-flight decode of it.
std::shared_ptr<Frame> get_frame(u32 luma) {
    std::unique_lock<std::mutex> l(g_frames_lock);
    // The guest is blocked on the VIC syncpoint while we decode (as it would be on hardware), so an
    // in-flight decode is worth waiting for; only a wedged decoder hits the bound.
    auto t0 = std::chrono::steady_clock::now();
    g_frames_cv.wait_for(l, std::chrono::milliseconds(1000),
                         [&] { return g_frames.count(luma) || !g_pending.count(luma); });
    double waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (waited > 5.0) g_stat_waits++;
    g_stat_max_wait = std::max(g_stat_max_wait, waited);
    auto it = g_frames.find(luma);
    std::shared_ptr<Frame> f;
    if (it != g_frames.end()) {
        f = it->second;
        g_stat_hits++;
    } else if (g_last_frame && g_last_frame->session == g_session.load()) {
        f = g_last_frame;  // same movie: repeating the previous frame is the least visible failure
        g_stat_fallback++;
    } else {
        g_stat_none++;
    }
    u32 total = g_stat_hits + g_stat_fallback + g_stat_none;
    if (total && total % 300 == 0) {
        hw_log("nvdec: frames %u: waits>5ms %u (max %.0f ms) repeated %u missing %u", total, g_stat_waits, g_stat_max_wait,
               g_stat_fallback, g_stat_none);
        g_stat_max_wait = 0;
    }
    return f;
}

// ------------------------------------------------------------------ bit writer (RBSP)
class BitWriter {
public:
    void bits(u32 v, int n) {
        for (int i = n - 1; i >= 0; i--) bit((v >> i) & 1);
    }
    void bit(u32 b) {
        cur_ = (u8)((cur_ << 1) | (b & 1));
        if (++n_ == 8) {
            out_.push_back(cur_);
            cur_ = 0;
            n_ = 0;
        }
    }
    void ue(u32 v) {
        u32 x = v + 1;
        int len = 0;
        while ((x >> len) > 1) len++;
        bits(0, len);
        bits(x, len + 1);
    }
    void se(s32 v) { ue(v <= 0 ? (u32)(-2 * v) : (u32)(2 * v - 1)); }
    void trailing() {
        bit(1);
        while (n_) bit(0);
    }
    // NAL unit with start code and emulation prevention.
    void emit_nal(std::vector<u8>& dst, u8 header) const {
        static const u8 sc[4] = {0, 0, 0, 1};
        dst.insert(dst.end(), sc, sc + 4);
        dst.push_back(header);
        int zeros = 0;
        for (u8 b : out_) {
            if (zeros >= 2 && b <= 3) {
                dst.push_back(3);
                zeros = 0;
            }
            dst.push_back(b);
            zeros = b == 0 ? zeros + 1 : 0;
        }
    }

private:
    std::vector<u8> out_;
    u8 cur_ = 0;
    int n_ = 0;
};

// nvdec H264 picture info ("H264DecoderContext"), offsets per the NVDEC firmware interface.
struct H264Info {
    u8 raw[0x2FC];
    u32 u32_at(u32 o) const { u32 v; memcpy(&v, raw + o, 4); return v; }
    s32 s32_at(u32 o) const { return (s32)u32_at(o); }
    u64 u64_at(u32 o) const { u64 v; memcpy(&v, raw + o, 8); return v; }
    u32 stream_len() const { return u32_at(0x48); }
    // parameter set at 0x58
    s32 log2_max_poc_lsb_minus4() const { return s32_at(0x58); }
    s32 delta_pic_order_always_zero() const { return s32_at(0x5C); }
    s32 frame_mbs_only() const { return s32_at(0x60); }
    u32 width_mbs() const { return u32_at(0x64); }
    u32 height_mbs() const { return u32_at(0x68); }
    u32 entropy_coding_mode() const { return u32_at(0x70); }
    s32 pic_order_present() const { return s32_at(0x74); }
    s32 num_ref_l0() const { return s32_at(0x78); }
    s32 num_ref_l1() const { return s32_at(0x7C); }
    s32 deblocking_control_present() const { return s32_at(0x80); }
    s32 redundant_pic_cnt_present() const { return s32_at(0x84); }
    u32 transform_8x8() const { return u32_at(0x88); }
    u32 luma_frame_offset() const { return u32_at(0x9C); }
    u64 flags() const { return u64_at(0xB0); }
    u32 flag(int b) const { return (u32)((flags() >> b) & 1); }
    u32 field(int lo, int n) const { return (u32)((flags() >> lo) & ((1ull << n) - 1)); }
    s32 sfield(int lo, int n) const {
        u32 v = field(lo, n);
        return (v & (1u << (n - 1))) ? (s32)v - (1 << n) : (s32)v;
    }
    const u8* scale4x4() const { return raw + 0x1C0; }
    const u8* scale8x8() const { return raw + 0x220; }
};

const u8 kZigzag4x4[16] = {0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15};
const u8 kZigzag8x8[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                           41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                           30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

void scaling_list(BitWriter& w, const u8* list, int n, const u8* zigzag) {
    int last = 8;
    for (int i = 0; i < n; i++) {
        int v = list[zigzag[i]];
        int delta = v - last;
        if (delta > 127) delta -= 256;
        if (delta < -128) delta += 256;
        w.se(delta);
        last = v;
    }
}

std::vector<u8> build_headers(const H264Info& p) {
    std::vector<u8> out;
    // ---- SPS (High profile)
    BitWriter s;
    s.bits(100, 8);  // profile_idc
    s.bits(0, 8);    // constraint flags
    s.bits(51, 8);   // level_idc
    s.ue(0);         // sps id
    s.ue(p.field(12, 2) ? p.field(12, 2) : 1);  // chroma_format_idc
    s.ue(0);         // bit_depth_luma_minus8
    s.ue(0);         // bit_depth_chroma_minus8
    s.bit(0);        // qpprime_y_zero_transform_bypass
    s.bit(0);        // seq_scaling_matrix_present (scaling lists go in the PPS)
    s.ue(p.field(8, 4));  // log2_max_frame_num_minus4
    u32 poc_type = p.field(14, 2);
    s.ue(poc_type);
    if (poc_type == 0) {
        s.ue((u32)p.log2_max_poc_lsb_minus4());
    } else if (poc_type == 1) {
        s.bit(p.delta_pic_order_always_zero() ? 1 : 0);
        s.se(0);
        s.se(0);
        s.ue(0);
    }
    s.ue(16);  // max_num_ref_frames
    s.bit(0);  // gaps_in_frame_num_allowed
    s.ue(p.width_mbs() - 1);
    u32 frame_mbs_only = p.frame_mbs_only() ? 1 : 0;
    s.ue(p.height_mbs() / (2 - frame_mbs_only) - 1);
    s.bit(frame_mbs_only);
    if (!frame_mbs_only) s.bit(p.flag(0));  // mb_adaptive_frame_field
    s.bit(p.flag(1));                       // direct_8x8_inference
    s.bit(0);                               // frame_cropping
    // VUI with only a bitstream restriction: NVDEC hands every picture to the guest as soon as it is
    // decoded (the game orders them for display by output buffer), so tell the host decoder there is
    // no reordering - otherwise Media Foundation buffers up to max_num_ref_frames pictures first.
    s.bit(1);   // vui_parameters_present
    s.bit(0);   // aspect_ratio_info_present
    s.bit(0);   // overscan_info_present
    s.bit(0);   // video_signal_type_present
    s.bit(0);   // chroma_loc_info_present
    s.bit(0);   // timing_info_present
    s.bit(0);   // nal_hrd_parameters_present
    s.bit(0);   // vcl_hrd_parameters_present
    s.bit(0);   // pic_struct_present
    s.bit(1);   // bitstream_restriction
    s.bit(1);   // motion_vectors_over_pic_boundaries
    s.ue(0);    // max_bytes_per_pic_denom
    s.ue(0);    // max_bits_per_mb_denom
    s.ue(16);   // log2_max_mv_length_horizontal
    s.ue(16);   // log2_max_mv_length_vertical
    s.ue(0);    // max_num_reorder_frames
    s.ue(16);   // max_dec_frame_buffering
    s.trailing();
    s.emit_nal(out, 0x67);

    // ---- PPS
    BitWriter q;
    q.ue(0);
    q.ue(0);
    q.bit(p.entropy_coding_mode() ? 1 : 0);
    q.bit(p.pic_order_present() ? 1 : 0);
    q.ue(0);  // slice groups
    q.ue((u32)std::max(0, p.num_ref_l0() - 1));
    q.ue((u32)std::max(0, p.num_ref_l1() - 1));
    q.bit(p.flag(2));          // weighted_pred
    q.bits(p.field(32, 2), 2); // weighted_bipred_idc
    q.se(p.sfield(16, 6));     // pic_init_qp_minus26
    q.se(0);                   // pic_init_qs_minus26
    q.se(p.sfield(22, 5));     // chroma_qp_index_offset
    q.bit(p.deblocking_control_present() ? 1 : 0);
    q.bit(p.flag(3));          // constrained_intra_pred
    q.bit(p.redundant_pic_cnt_present() ? 1 : 0);
    q.bit(p.transform_8x8() ? 1 : 0);
    q.bit(1);  // pic_scaling_matrix_present
    for (int i = 0; i < 6; i++) {
        q.bit(1);
        scaling_list(q, p.scale4x4() + i * 16, 16, kZigzag4x4);
    }
    if (p.transform_8x8()) {
        for (int i = 0; i < 2; i++) {
            q.bit(1);
            scaling_list(q, p.scale8x8() + i * 64, 64, kZigzag8x8);
        }
    }
    q.se(p.sfield(27, 5));  // second_chroma_qp_index_offset
    q.trailing();
    q.emit_nal(out, 0x68);
    return out;
}

// ------------------------------------------------------------------ Media Foundation H.264 decoder
class MfDecoder {
public:
    bool ok() const { return mft_ != nullptr; }

    MfDecoder() {
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(MFStartup(MF_VERSION, MFSTARTUP_LITE))) {
            hw_log("nvdec: MFStartup failed");
            return;
        }
        if (FAILED(CoCreateInstance(CLSID_CMSH264DecoderMFT, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&mft_)))) {
            hw_log("nvdec: H.264 decoder MFT not available");
            mft_ = nullptr;
            return;
        }
        IMFAttributes* attr = nullptr;
        if (SUCCEEDED(mft_->GetAttributes(&attr)) && attr) {
            attr->SetUINT32(CODECAPI_AVLowLatencyMode, TRUE);
            attr->SetUINT32(MF_LOW_LATENCY, TRUE);
            attr->Release();
        }
        IMFMediaType* in = nullptr;
        MFCreateMediaType(&in);
        in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        in->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
        HRESULT hr = mft_->SetInputType(0, in, 0);
        in->Release();
        if (FAILED(hr)) {
            hw_log("nvdec: SetInputType failed 0x%08lx", hr);
            mft_->Release();
            mft_ = nullptr;
            return;
        }
        set_output_type();
        mft_->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
        mft_->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
        hw_log("nvdec: Media Foundation H.264 decoder ready");
    }

    // Reuse for a new stream: drop buffered pictures, keep the (expensive to create) transform.
    void restart() {
        if (!mft_) return;
        mft_->ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0);
        mft_->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
    }
    // Decode one access unit; decoded pictures are delivered through put_frame keyed by their tag.
    void decode(const std::vector<u8>& au, u32 tag) {
        if (!mft_) return;
        IMFSample* sample = nullptr;
        IMFMediaBuffer* buf = nullptr;
        MFCreateSample(&sample);
        MFCreateMemoryBuffer((DWORD)au.size(), &buf);
        BYTE* p = nullptr;
        buf->Lock(&p, nullptr, nullptr);
        memcpy(p, au.data(), au.size());
        buf->Unlock();
        buf->SetCurrentLength((DWORD)au.size());
        sample->AddBuffer(buf);
        sample->SetSampleTime((LONGLONG)tag * 10);
        sample->SetSampleDuration(10);
        HRESULT hr = mft_->ProcessInput(0, sample, 0);
        if (hr == MF_E_NOTACCEPTING) {
            drain();
            hr = mft_->ProcessInput(0, sample, 0);
        }
        if (FAILED(hr)) {
            static int warned = 0;
            if (warned++ < 4) hw_log("nvdec: ProcessInput failed 0x%08lx", hr);
        }
        sample->Release();
        buf->Release();
        drain();
    }

private:
    void set_output_type() {
        for (DWORD i = 0;; i++) {
            IMFMediaType* t = nullptr;
            if (FAILED(mft_->GetOutputAvailableType(0, i, &t))) break;
            GUID sub{};
            t->GetGUID(MF_MT_SUBTYPE, &sub);
            if (sub == MFVideoFormat_NV12) {
                mft_->SetOutputType(0, t, 0);
                UINT32 w = 0, h = 0;
                MFGetAttributeSize(t, MF_MT_FRAME_SIZE, &w, &h);
                width_ = w;
                height_ = h;
                UINT32 stride = 0;
                stride_ = SUCCEEDED(t->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride)) ? (s32)stride : (s32)w;
                t->Release();
                return;
            }
            t->Release();
        }
    }

    void drain() {
        for (;;) {
            MFT_OUTPUT_STREAM_INFO si{};
            mft_->GetOutputStreamInfo(0, &si);
            MFT_OUTPUT_DATA_BUFFER out{};
            IMFSample* s = nullptr;
            if (!(si.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES))) {
                IMFMediaBuffer* b = nullptr;
                MFCreateSample(&s);
                MFCreateMemoryBuffer(si.cbSize ? si.cbSize : 4 << 20, &b);
                s->AddBuffer(b);
                b->Release();
                out.pSample = s;
            }
            DWORD status = 0;
            HRESULT hr = mft_->ProcessOutput(0, 1, &out, &status);
            if (out.pEvents) out.pEvents->Release();
            if (hr == MF_E_TRANSFORM_STREAM_CHANGE) {
                if (s) s->Release();
                set_output_type();
                continue;
            }
            if (FAILED(hr)) {
                if (s) s->Release();
                break;
            }
            IMFSample* got = out.pSample;
            deliver(got);
            if (got) got->Release();
        }
    }

    void deliver(IMFSample* s) {
        if (!s || !width_ || !height_) return;
        LONGLONG t = 0;
        s->GetSampleTime(&t);
        u32 tag = (u32)(t / 10);
        IMFMediaBuffer* b = nullptr;
        if (FAILED(s->ConvertToContiguousBuffer(&b))) return;
        BYTE* p = nullptr;
        DWORD len = 0;
        if (SUCCEEDED(b->Lock(&p, nullptr, &len))) {
            auto f = std::make_shared<Frame>();
            f->width = width_;
            f->height = height_;
            u32 stride = stride_ > 0 ? (u32)stride_ : width_;
            // NV12 with the chroma plane after `height` (possibly aligned) rows of luma.
            u32 luma_rows = height_;
            if ((u64)stride * luma_rows * 3 / 2 > len) luma_rows = (u32)(len * 2 / 3 / stride);
            u32 uv_rows_at = (u32)std::max<u64>(luma_rows, (len / stride) * 2 / 3);
            f->y.resize((size_t)width_ * height_);
            f->uv.resize((size_t)width_ * (height_ / 2));
            for (u32 y = 0; y < height_ && (u64)(y + 1) * stride <= len; y++)
                memcpy(&f->y[(size_t)y * width_], p + (size_t)y * stride, width_);
            for (u32 y = 0; y < height_ / 2; y++) {
                u64 o = (u64)(uv_rows_at + y) * stride;
                if (o + width_ > len) break;
                memcpy(&f->uv[(size_t)y * width_], p + o, width_);
            }
            b->Unlock();
            put_frame(tag, f);
            static std::atomic<int> n{0};
            if (n++ == 0) hw_log("nvdec: first frame decoded (%ux%u)", width_, height_);
        }
        b->Release();
    }

    IMFTransform* mft_ = nullptr;
    u32 width_ = 0, height_ = 0;
    s32 stride_ = 0;
};

// One decoder per NVDEC channel state, owned and driven by the decode worker thread so the guest
// thread that submits never blocks on the codec.
std::map<host1x::ChannelState*, std::unique_ptr<MfDecoder>> g_decoders;
std::unique_ptr<MfDecoder> g_spare_decoder;  // warm transform for the next movie (creation costs ~200 ms)
std::atomic<bool> g_new_session_pending{false};
struct DecodeJob {
    host1x::ChannelState* st;
    std::vector<u8> au;
    u32 tag;
};
std::mutex g_jobs_lock;
std::condition_variable g_jobs_cv;
std::deque<DecodeJob> g_jobs;

void decode_worker() {
    SetThreadDescription(GetCurrentThread(), L"HWDER nvdec");
    g_spare_decoder = std::make_unique<MfDecoder>();  // warm up before the first movie asks
    for (;;) {
        DecodeJob job;
        {
            std::unique_lock<std::mutex> l(g_jobs_lock);
            g_jobs_cv.wait(l, [] { return !g_jobs.empty(); });
            job = std::move(g_jobs.front());
            g_jobs.pop_front();
        }
        if (g_new_session_pending.exchange(false)) {
            // New movie: recycle the previous stream's decoder instead of creating a new one.
            for (auto& [st, dec] : g_decoders)
                if (dec && !g_spare_decoder) g_spare_decoder = std::move(dec);
            g_decoders.clear();
            if (g_spare_decoder) g_spare_decoder->restart();
        }
        auto& d = g_decoders[job.st];
        if (!d) d = g_spare_decoder ? std::move(g_spare_decoder) : std::make_unique<MfDecoder>();
        d->decode(job.au, job.tag);
        {
            std::lock_guard<std::mutex> l(g_frames_lock);
            auto it = g_pending.find(job.tag);
            if (it != g_pending.end()) g_pending.erase(it);
        }
        g_frames_cv.notify_all();
    }
}

static std::once_flag g_worker_once;
void start_decode_worker() {
    std::call_once(g_worker_once, [] { std::thread(decode_worker).detach(); });
}
void queue_decode(DecodeJob&& job) {
    start_decode_worker();
    {
        std::lock_guard<std::mutex> l(g_frames_lock);
        u64 now = GetTickCount64();
        if (g_last_job_ms && now - g_last_job_ms > 400) {  // a new movie: forget the old one's frames
            g_session++;
            g_frames.clear();
            g_last_frame.reset();
            g_new_session_pending = true;
            hw_log("nvdec: new decode session %u", g_session.load());
        }
        g_last_job_ms = now;
        g_pending.insert(job.tag);
    }
    {
        std::lock_guard<std::mutex> l(g_jobs_lock);
        g_jobs.push_back(std::move(job));
    }
    g_jobs_cv.notify_one();
}

// ------------------------------------------------------------------ YUV -> RGB
inline u8 clamp8(int v) { return (u8)(v < 0 ? 0 : v > 255 ? 255 : v); }

// Block-linear (GOB 64x8) byte offset of (x_bytes, y) in a surface `width_bytes` wide.
inline u64 bl_offset(u32 x, u32 y, u32 width_bytes, u32 bh_log2) {
    u32 gobs_h = 1u << bh_log2;
    u32 block_rows = 8 * gobs_h;
    u32 blocks_per_row = (width_bytes + 63) / 64;
    u64 block_size = 512ull * gobs_h;
    u64 off = (u64)(y / block_rows) * blocks_per_row * block_size + (u64)(x / 64) * block_size +
              (u64)((y % block_rows) / 8) * 512;
    off += ((x % 64) / 32) * 256 + ((y % 8) / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + (x % 16);
    return off;
}

}  // namespace
void prewarm_decoder() { start_decode_worker(); }

void nvdec_execute(host1x::ChannelState& st) {
    u32 codec = (u32)st.regs[0x80];
    if (codec != 3) {  // H.264 only
        static int warned = 0;
        if (warned++ < 2) hw_log("nvdec: codec %u not supported", codec);
        return;
    }
    H264Info info;
    if (!read_iova((u32)(st.regs[0x101] << 8), info.raw, sizeof(info.raw))) return;
    u32 len = info.stream_len();
    const u8* bits = iova_ptr((u32)(st.regs[0x102] << 8), len);
    if (!bits || !len) return;
    std::vector<u8> au = build_headers(info);
    au.insert(au.end(), bits, bits + len);
    u32 pic = info.field(34, 7);
    u32 luma = (u32)((st.regs[0x10C + (pic < 17 ? pic : 0)] << 8) + ((u64)info.luma_frame_offset() << 8));

    queue_decode({&st, std::move(au), luma});
}

void vic_execute(host1x::ChannelState& st) {
    // VIC registers (u32 method index): surfaces[slot][8][3] at 0x100, config struct 0x1C2, output 0x1C8/9.
    u8 cfg[0x610];
    if (!read_iova((u32)(st.regs[0x1C2] << 8), cfg, sizeof(cfg))) return;
    u32 osc0, osc1, osc2;
    memcpy(&osc0, cfg + 0x20, 4);
    memcpy(&osc1, cfg + 0x24, 4);
    memcpy(&osc2, cfg + 0x28, 4);
    u32 fmt = osc0 & 0x7F, kind = (osc0 >> 11) & 0xF, bh = (osc0 >> 15) & 0xF;
    u32 sw = (osc1 & 0x3FFF) + 1, sh = ((osc1 >> 14) & 0x3FFF) + 1;
    u32 lw = (osc2 & 0x3FFF) + 1, lh = ((osc2 >> 14) & 0x3FFF) + 1;
    u32 w = std::min(sw, lw), h = std::min(sh, lh);

    // First enabled slot provides the picture.
    std::shared_ptr<Frame> f;
    for (int slot = 0; slot < 8 && !f; slot++) {
        u64 sc;
        memcpy(&sc, cfg + 0x90 + slot * 0xB0, 8);
        if (!(sc & 1)) continue;
        u32 luma = (u32)(st.regs[0x100 + slot * 24 + 0] << 8);
        f = get_frame(luma);
    }
    if (!f) {
        // Nothing decoded yet for this movie: compose black rather than leaving the previous movie's
        // picture in the output buffer.
        static std::shared_ptr<Frame> black;
        if (!black || black->width != w || black->height != h) {
            black = std::make_shared<Frame>();
            black->width = w;
            black->height = h;
            black->y.assign((size_t)w * h, 16);
            black->uv.assign((size_t)w * (h / 2 + 1), 128);
        }
        f = black;
    }
    // Coded size is macroblock aligned (e.g. 1088 for 1080p): crop instead of scaling.
    auto src = std::make_shared<Frame>(*f);
    if (f->height >= h && f->height - h < 16) src->height = h;
    if (f->width >= w && f->width - w < 16) src->width = w;
    u32 fw = f->width;
    f = src;

    u32 out_luma = (u32)(st.regs[0x1C8] << 8), out_chroma = (u32)(st.regs[0x1C9] << 8);
    bool argb = fmt == 32;  // A8R8G8B8: B,G,R,A byte order; A8B8G8R8/X8B8G8R8: R,G,B,A
    bool hd = f->height >= 720;
    // BT.709 / BT.601 limited range, fixed point 8.8
    int cy = 298, crv = hd ? 459 : 409, cgu = hd ? 55 : 100, cgv = hd ? 136 : 208, cbu = hd ? 541 : 516;

    if (fmt == 31 || fmt == 32 || fmt == 35) {
        u32 stride = (lw * 4 + 15) & ~15u;
        u64 size = kind ? (u64)((lw * 4 + 63) / 64) * 64 * ((lh + (8u << bh) - 1) & ~((8u << bh) - 1)) : (u64)stride * lh;
        u8* dst = iova_ptr(out_luma, size);
        if (!dst) return;
        for (u32 y = 0; y < h; y++) {
            u32 fy = std::min(y * f->height / h, f->height - 1);
            const u8* yl = &f->y[(size_t)fy * fw];
            const u8* uvl = &f->uv[(size_t)(fy / 2) * fw];
            for (u32 x = 0; x < w; x++) {
                u32 fx = std::min(x * f->width / w, f->width - 1);
                int Y = (yl[fx] - 16) * cy, U = uvl[fx & ~1u] - 128, V = uvl[(fx & ~1u) + 1] - 128;
                u8 r = clamp8((Y + crv * V + 128) >> 8);
                u8 g = clamp8((Y - cgu * U - cgv * V + 128) >> 8);
                u8 b = clamp8((Y + cbu * U + 128) >> 8);
                u8* px = dst + (kind ? bl_offset(x * 4, y, lw * 4, bh) : (u64)y * stride + x * 4);
                if (argb) { px[0] = b; px[1] = g; px[2] = r; }
                else { px[0] = r; px[1] = g; px[2] = b; }
                px[3] = 0xFF;
            }
        }
        gpu::renderer()->invalidate_region((u64)dst, size);
    } else if (fmt == 68 || fmt == 67) {  // Y8__V8U8_N420 / Y8__U8V8_N420 (NV12-like)
        u32 stride = (lw + 255) & ~255u;
        u8* yd = iova_ptr(out_luma, (u64)stride * lh);
        u8* cd = iova_ptr(out_chroma, (u64)stride * (lh / 2));
        if (!yd || !cd) return;
        for (u32 y = 0; y < h; y++) {
            u32 fy = std::min(y * f->height / h, f->height - 1);
            for (u32 x = 0; x < w; x++) {
                u32 fx = std::min(x * f->width / w, f->width - 1);
                yd[kind ? bl_offset(x, y, lw, bh) : (u64)y * stride + x] = f->y[(size_t)fy * fw + fx];
            }
        }
        for (u32 y = 0; y < h / 2; y++) {
            u32 fy = std::min(y * (f->height / 2) / (h / 2), f->height / 2 - 1);
            for (u32 x = 0; x < w / 2; x++) {
                u32 fx = std::min(x * (f->width / 2) / (w / 2), f->width / 2 - 1);
                const u8* uv = &f->uv[(size_t)fy * fw + fx * 2];
                u8* o = cd + (kind ? bl_offset(x * 2, y, lw, bh) : (u64)y * stride + x * 2);
                // NV naming lists components MSB first: Y8__V8U8 is byte0 = U, byte1 = V (plain NV12).
                if (fmt == 68) { o[0] = uv[0]; o[1] = uv[1]; }
                else { o[0] = uv[1]; o[1] = uv[0]; }
            }
        }
        gpu::renderer()->invalidate_region((u64)yd, (u64)stride * lh);
        gpu::renderer()->invalidate_region((u64)cd, (u64)stride * (lh / 2));
    } else {
        static int warned = 0;
        if (warned++ < 2) hw_log("vic: output format %u not supported", fmt);
    }
    static std::atomic<int> n{0};
    if (n++ == 0) hw_log("vic: first frame composed (%ux%u fmt %u kind %u)", w, h, fmt, kind);
}

}  // namespace video
