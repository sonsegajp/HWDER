// Audio: audren:u (the audio renderer: voices -> mixes -> sink), audout:u, and WASAPI output.
//
// The renderer runs on its own thread at the Switch's rate (one frame = sample_count samples,
// 5 ms at 48 kHz). The game drives it through RequestUpdate: a blob of per-object input
// parameters (behavior, memory pools, voice resources, voices, effects, splitters, mixes, sinks,
// performance) answered with a blob of per-object statuses. Layouts follow the NX audio
// renderer protocol (revision given by the game; this SDK uses an early one).
#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>

#include "service/common.h"

#pragma comment(lib, "ole32.lib")

namespace ipc {
namespace {

constexpr u32 kMaxMixBuffers = 24, kMaxChannels = 6, kMaxWaveBuffers = 4, kTargetRate = 48000;

static u32 revision_num(u32 rev) { return rev >= 0x100 ? (rev - 0x30564552u) >> 24 : rev; }  // 'REV0'

// ------------------------------------------------------------------ host output (WASAPI)
// Single-producer / single-consumer ring of interleaved stereo s16 at 48 kHz.
struct OutputRing {
    static constexpr u32 kFrames = 48000 / 4;  // 250 ms
    s16 data[kFrames * 2];
    std::atomic<u32> write{0}, read{0};
    u32 fill() const { return (write.load() - read.load()) % kFrames; }
    void push(const s16* frames, u32 n) {
        u32 w = write.load();
        if (fill() + n >= kFrames - 1) return;  // consumer stalled or absent: drop
        for (u32 i = 0; i < n; i++) {
            data[(w % kFrames) * 2] = frames[i * 2];
            data[(w % kFrames) * 2 + 1] = frames[i * 2 + 1];
            w++;
        }
        write.store(w);
    }
    u32 pop(s16* out, u32 n) {
        u32 r = read.load(), avail = fill();
        u32 got = std::min(n, avail);
        for (u32 i = 0; i < got; i++) {
            out[i * 2] = data[(r % kFrames) * 2];
            out[i * 2 + 1] = data[(r % kFrames) * 2 + 1];
            r++;
        }
        read.store(r);
        return got;
    }
};
static OutputRing g_ring;

static void wasapi_thread() {
    SetThreadDescription(GetCurrentThread(), L"HWDER audio out");
    if (getenv("HWDER_MUTE")) {  // test runs: never open the output device
        hw_log("audio: muted (HWDER_MUTE) - output disabled");
        for (;;) {
            s16 sink[1024];
            g_ring.pop(sink, 256);
            Sleep(5);
        }
    }
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IMMDeviceEnumerator* en = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&en))) {
        hw_log("audio: no WASAPI - output disabled");
        return;
    }
    for (;;) {  // reopen on device loss
        IMMDevice* dev = nullptr;
        IAudioClient* client = nullptr;
        IAudioRenderClient* render = nullptr;
        if (FAILED(en->GetDefaultAudioEndpoint(eRender, eConsole, &dev))) {
            Sleep(1000);
            continue;
        }
        WAVEFORMATEX fmt{};
        fmt.wFormatTag = WAVE_FORMAT_PCM;
        fmt.nChannels = 2;
        fmt.nSamplesPerSec = kTargetRate;
        fmt.wBitsPerSample = 16;
        fmt.nBlockAlign = 4;
        fmt.nAvgBytesPerSec = kTargetRate * 4;
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        bool ok = SUCCEEDED(dev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&client)) &&
                  SUCCEEDED(client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                               AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                                                   AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                                               200000, 0, &fmt, nullptr)) &&
                  SUCCEEDED(client->SetEventHandle(ev)) &&
                  SUCCEEDED(client->GetService(__uuidof(IAudioRenderClient), (void**)&render)) &&
                  SUCCEEDED(client->Start());
        UINT32 buffer_frames = 0;
        if (ok) client->GetBufferSize(&buffer_frames);
        if (ok) hw_log("audio: WASAPI output open (%u frame buffer)", buffer_frames);
        else hw_log("audio: WASAPI init failed - retrying");
        while (ok) {
            if (WaitForSingleObject(ev, 2000) != WAIT_OBJECT_0) break;
            UINT32 padding = 0;
            if (FAILED(client->GetCurrentPadding(&padding))) break;
            UINT32 n = buffer_frames - padding;
            if (!n) continue;
            BYTE* p = nullptr;
            if (FAILED(render->GetBuffer(n, &p))) break;
            u32 got = g_ring.pop((s16*)p, n);
            if (got < n) memset(p + got * 4, 0, (n - got) * 4);
            static const bool mute = getenv("HWDER_MUTE") != nullptr;  // silent output (test runs)
            if (mute) memset(p, 0, (size_t)n * 4);
            render->ReleaseBuffer(n, 0);
        }
        if (render) render->Release();
        if (client) client->Release();
        if (dev) dev->Release();
        CloseHandle(ev);
        Sleep(500);
    }
}

// ------------------------------------------------------------------ renderer
struct Params {
    u32 sample_rate, sample_count, mixes, sub_mixes, voices, sinks, effects, perf_frames;
    u8 voice_drop, unk21, device, exec_mode;
    u32 splitter_infos;
    s32 splitter_destinations;
    u32 external_context_size, revision;
};
static_assert(sizeof(Params) == 0x34);

#pragma pack(push, 1)
struct WaveBufferIn {
    u64 address, size;
    s32 start_offset, end_offset;
    u8 loop, stream_ended, sent_to_dsp, pad;
    s32 loop_count;
    u64 context_address, context_size;
    u32 loop_start, loop_end;
};
static_assert(sizeof(WaveBufferIn) == 0x38);

struct VoiceIn {  // version 1 (revisions < 15)
    u32 id, node_id;
    u8 is_new, in_use, play_state, sample_format;
    u32 sample_rate;
    s32 priority, sort_order;
    u32 channel_count;
    float pitch, volume;
    u8 biquads[0x18];
    u32 wave_buffer_count;
    u16 wave_buffer_index;
    u8 unk42[6];
    u64 src_data_address, src_data_size;
    u32 mix_id, splitter_id;
    WaveBufferIn wave[kMaxWaveBuffers];
    u32 channel_resource_ids[kMaxChannels];
    u8 clear_voice_drop, flush_buffer_count;
    u8 unk15A[2];
    u8 flags, unk15D, src_quality;
    u8 unk15F[0x11];
};
static_assert(sizeof(VoiceIn) == 0x170);

struct VoiceOut {
    u64 played_sample_count;
    u32 wave_buffers_consumed;
    u8 voice_dropped;
    u8 pad[3];
};
struct ResourceIn {
    u32 id;
    float mix_volumes[kMaxMixBuffers];
    u8 in_use;
    u8 pad[0xB];
};
static_assert(sizeof(ResourceIn) == 0x70);
struct MixIn {
    float volume;
    u32 sample_rate, buffer_count;
    u8 in_use, is_dirty;
    u8 pad0[2];
    s32 mix_id;
    u32 effect_count;
    s32 node_id;
    u8 unk1C[8];
    float mix_volumes[kMaxMixBuffers][kMaxMixBuffers];
    s32 dest_mix_id, dest_splitter_id;
    u8 pad[4];
};
static_assert(sizeof(MixIn) == 0x930);
struct SinkIn {
    u8 type, in_use;
    u8 pad[2];
    u32 node_id;
    u8 unk08[0x18];
    union {
        struct {
            char name[0x100];
            u32 input_count;
            s8 inputs[kMaxChannels];
            u8 unk10A, downmix_enabled;
            float downmix_coeff[4];
        } device;
        struct {
            u64 cpu_address;
            u32 size, input_count, sample_count, previous_pos;
            u32 format;
            s8 inputs[kMaxChannels];
            u8 in_use;
            u8 unk23[5];
        } circular;
    };
    u8 pad_end[4];
};
static_assert(sizeof(SinkIn) == 0x140);
struct PoolIn {
    u64 address, size;
    u32 state;
    u8 in_use;
    u8 pad[0xB];
};
static_assert(sizeof(PoolIn) == 0x20);
struct EffectIn {
    u8 type, is_new, enabled, pad;
    u32 mix_id;
    u64 workbuffer, workbuffer_size;
    u32 process_order;
    u8 unk1C[4];
    u8 specific[0xA0];
};
static_assert(sizeof(EffectIn) == 0xC0);
struct UpdateHeader {
    u32 revision, behavior, pools, voices, resources, effects, mixes, sinks, perf, splitter, render_info;
    u8 unk2C[0x10];
    u32 size;
};
static_assert(sizeof(UpdateHeader) == 0x40);
#pragma pack(pop)

struct VoiceState {
    VoiceIn in{};
    bool active = false;
    // wave buffer slots as the DSP sees them
    struct Slot {
        bool valid = false;
        WaveBufferIn wb{};
    } slots[kMaxWaveBuffers];
    u32 index = 0;        // slot being played
    s64 pos = 0;          // sample position in the current buffer
    s32 loops_left = 0;
    bool stopped_output = false;
    u64 played = 0;
    u32 consumed = 0;
    // resampler: last two source samples per channel
    float prev[kMaxChannels][2] = {};
    double frac = 0;
    // ADPCM
    s16 yn1 = 0, yn2 = 0;
    s16 coeffs[16] = {};
};

struct MixState {
    MixIn in{};
    u32 buffer_offset = 0;
};

static s16 clamp16(s32 v) { return (s16)std::clamp(v, -32768, 32767); }

class Renderer {
public:
    explicit Renderer(const Params& p) : params_(p), rev_(revision_num(p.revision)) {
        voices_.resize(p.voices);
        resources_.resize(p.voices);
        mixes_.resize(p.sub_mixes + 1);
        sinks_.resize(p.sinks);
        pools_.resize(p.effects + p.voices * kMaxWaveBuffers);
        effects_.resize(p.effects);
        splitters_.resize(p.splitter_infos);
        splitter_dests_.resize(std::max(p.splitter_destinations, 0));
        event_ = make_event();
        u32 total = p.mixes + (p.sub_mixes + 1) * kMaxMixBuffers;
        mix_buffers_.assign((size_t)total * p.sample_count, 0.0f);
        hw_log("audren: renderer %u Hz x%u, %u voices, %u submixes, %u sinks, %u effects, rev %u", p.sample_rate,
               p.sample_count, p.voices, p.sub_mixes, p.sinks, p.effects, rev_);
    }

    std::shared_ptr<ReadableEvent> event() { return event_; }
    const Params& params() const { return params_; }
    bool started() const { return started_; }
    void start() { started_ = true; }
    void stop() { started_ = false; }

    // One RequestUpdate: parse input sections in order, write statuses.
    Result update(const u8* in, u64 in_size, u8* out, u64 out_size) {
        std::lock_guard<std::mutex> l(m_);
        if (in_size < sizeof(UpdateHeader)) return ModuleResult(153, 4);
        UpdateHeader h;
        memcpy(&h, in, sizeof(h));
        const u8* p = in + sizeof(h);
        const u8* end = in + in_size;
        auto section = [&](u32 size) -> const u8* {
            const u8* s = p;
            p += size;
            return p <= end ? s : nullptr;
        };
        memset(out, 0, std::min<u64>(out_size, sizeof(UpdateHeader)));
        u8* o = out + sizeof(UpdateHeader);
        UpdateHeader oh{};
        oh.revision = h.revision;

        section(h.behavior);  // BehaviorInfo::InParameter (flags) - nothing to do

        // memory pools: state machine only (guest memory is directly accessible)
        if (const u8* s = section(h.pools)) {
            u32 n = h.pools / sizeof(PoolIn);
            for (u32 i = 0; i < n && i < pools_.size(); i++) {
                PoolIn pi;
                memcpy(&pi, s + i * sizeof(PoolIn), sizeof(pi));
                u32 st = pi.state;
                if (st == 4) st = 5;       // RequestAttach -> Attached
                else if (st == 2) st = 3;  // RequestDetach -> Detached
                pools_[i] = st;
                u32 os[4] = {st, 0, 0, 0};
                memcpy(o, os, 16);
                o += 16;
            }
            oh.pools = n * 16;
        }
        if (const u8* s = section(h.resources)) {
            u32 n = h.resources / sizeof(ResourceIn);
            for (u32 i = 0; i < n && i < resources_.size(); i++) memcpy(&resources_[i], s + i * sizeof(ResourceIn), sizeof(ResourceIn));
        }
        if (const u8* s = section(h.voices)) {
            u32 n = h.voices / sizeof(VoiceIn);
            for (u32 i = 0; i < n && i < voices_.size(); i++) {
                VoiceIn vi;
                memcpy(&vi, s + i * sizeof(VoiceIn), sizeof(vi));
                update_voice(voices_[i], vi);
                VoiceOut vo{voices_[i].played, voices_[i].consumed, 0, {}};
                memcpy(o, &vo, sizeof(vo));
                o += sizeof(vo);
            }
            oh.voices = n * sizeof(VoiceOut);
        }
        if (const u8* s = section(h.effects)) {
            u32 n = h.effects / sizeof(EffectIn);
            u32 out_sz = rev_ >= 9 ? 0x90 : 0x10;
            for (u32 i = 0; i < n && i < effects_.size(); i++) {
                memcpy(&effects_[i], s + i * sizeof(EffectIn), sizeof(EffectIn));
                if (effects_[i].is_new) {
                    const u8* sp = effects_[i].specific;
                    hw_log("audren: effect %u type %u mix %u enabled %u in[%d %d %d %d %d %d] out[%d %d %d %d %d %d]", i, effects_[i].type,
                           effects_[i].mix_id, effects_[i].enabled, (s8)sp[0], (s8)sp[1], (s8)sp[2], (s8)sp[3], (s8)sp[4], (s8)sp[5], (s8)sp[6],
                           (s8)sp[7], (s8)sp[8], (s8)sp[9], (s8)sp[10], (s8)sp[11]);
                }
                memset(o, 0, out_sz);
                o[0] = effects_[i].enabled ? 3 : 4;  // Used : Removed
                o += out_sz;
            }
            oh.effects = n * out_sz;
        }
        if (rev_ >= 2) {
            if (const u8* s = section(h.splitter)) {  // SNDH header, SNDI infos (id, count, dest ids), SNDD destinations
                const u8* q = s;
                const u8* end = s + h.splitter;
                u32 magic = 0;
                if (end - q >= 16) memcpy(&magic, q, 4);
                if (magic == 0x48444E53u) {
                    s32 ic = 0, dc = 0;
                    memcpy(&ic, q + 4, 4);
                    memcpy(&dc, q + 8, 4);
                    q += 16;
                    for (s32 i = 0; i < ic && end - q >= 16; i++) {
                        s32 id = 0, cnt = 0;
                        memcpy(&id, q + 4, 4);
                        memcpy(&cnt, q + 8, 4);
                        q += 16;
                        if (id >= 0 && (size_t)id < splitters_.size()) {
                            splitters_[id].clear();
                            for (s32 k = 0; k < cnt && end - q >= 4; k++) {
                                s32 d;
                                memcpy(&d, q, 4);
                                splitters_[id].push_back(d);
                                q += 4;
                            }
                        } else {
                            q += 4 * std::max(cnt, 0);
                        }
                    }
                    for (s32 i = 0; i < dc && end - q >= 0x70; i++) {
                        s32 id = 0;
                        memcpy(&id, q + 4, 4);
                        if (id >= 0 && (size_t)id < splitter_dests_.size()) {
                            SplitterDest& d = splitter_dests_[id];
                            memcpy(d.mix_volumes, q + 8, sizeof(d.mix_volumes));
                            memcpy(&d.dest_mix_id, q + 8 + 24 * 4, 4);
                            d.in_use = q[8 + 24 * 4 + 4] != 0;
                        }
                        q += 0x70;
                    }
                }
            }
        }
        if (const u8* s = section(h.mixes)) {
            const u8* q = s;
            u32 n = h.mixes;
            if (rev_ >= 7) {  // dirty-only update: 0x20 header {magic, count} then `count` MixIn
                u32 count = *(const u32*)(q + 4);
                q += 0x20;
                n = count * sizeof(MixIn);
            }
            u32 count = n / sizeof(MixIn);
            for (u32 i = 0; i < count; i++) {
                MixIn mi;
                memcpy(&mi, q + i * sizeof(MixIn), sizeof(mi));
                u32 idx = rev_ >= 7 ? (mi.mix_id == -1 ? 0 : (u32)mi.mix_id) : i;
                if (idx < mixes_.size()) mixes_[idx].in = mi;
            }
            u32 off = 0;
            for (auto& m : mixes_) {
                m.buffer_offset = off;
                if (m.in.in_use) off += m.in.buffer_count;
            }
        }
        if (const u8* s = section(h.sinks)) {
            u32 n = h.sinks / sizeof(SinkIn);
            for (u32 i = 0; i < n && i < sinks_.size(); i++) {
                memcpy(&sinks_[i], s + i * sizeof(SinkIn), sizeof(SinkIn));
                memset(o, 0, 0x20);
                *(u32*)o = sinks_[i].type == 2 ? circ_pos_ : 0;
                o += 0x20;
            }
            oh.sinks = n * 0x20;
        }
        if (section(h.perf)) {
            memset(o, 0, 0x10);
            o += 0x10;
            oh.perf = 0x10;
        }
        // behaviour out: error list (none)
        memset(o, 0, 0xB0);
        o += 0xB0;
        oh.behavior = 0xB0;
        if (rev_ >= 5) {
            u64 ri[2] = {frames_, 0};
            memcpy(o, ri, 16);
            o += 16;
            oh.render_info = 16;
        }
        oh.size = (u32)(o - out);
        memcpy(out, &oh, sizeof(oh));
        return 0;
    }

    // Render one frame into `out` (stereo s16 at params.sample_rate).
    void render(s16* out) {
        std::lock_guard<std::mutex> l(m_);
        u32 n = params_.sample_count;
        std::fill(mix_buffers_.begin(), mix_buffers_.end(), 0.0f);
        for (auto& v : voices_) render_voice(v);
        // Effects of a mix run after its buffers are filled and before it feeds its destination.
        // BufferMixer (type 1) is real routing: games fold sound categories from their input buffers
        // into the output buffers with it, so without it those sounds never reach the sink. The
        // time/frequency effects are passed through dry (input -> output) rather than dropped.
        auto run_effects = [&](u32 mix_id) {
            MixState& m = mixes_[mix_id];
            for (const EffectIn& e : effects_) {
                if (!e.enabled || e.mix_id != mix_id) continue;
                const u8* sp = e.specific;
                switch (e.type) {
                case 1: {  // BufferMixer: s8 inputs[6], s8 outputs[6], float volumes[6], u32 count
                    u32 count = 0;
                    memcpy(&count, sp + 36, 4);
                    for (u32 c = 0; c < std::min<u32>(count, 6); c++) {
                        s8 in = (s8)sp[c], out_ = (s8)sp[6 + c];
                        float vol;
                        memcpy(&vol, sp + 12 + c * 4, 4);
                        if (in < 0 || out_ < 0 || (u32)in >= m.in.buffer_count || (u32)out_ >= m.in.buffer_count || in == out_ || vol == 0) continue;
                        float* src = buf(m.buffer_offset + in);
                        float* dst = buf(m.buffer_offset + out_);
                        for (u32 k = 0; k < n; k++) dst[k] += src[k] * vol;
                    }
                    break;
                }
                default: break;  // Aux/Delay/Reverb/I3DL2/Biquad/Limiter: in-place or send/return - the dry mix flows on

                }
            }
        };
        // submixes feed their destination (process higher ids first: leaves tend to be allocated later)
        for (size_t i = mixes_.size(); i-- > 1;) {
            MixState& m = mixes_[i];
            if (!m.in.in_use) continue;
            run_effects((u32)i);
            if (m.in.dest_mix_id < 0 || (u32)m.in.dest_mix_id >= mixes_.size()) continue;
            MixState& d = mixes_[m.in.dest_mix_id];
            for (u32 s = 0; s < m.in.buffer_count; s++)
                for (u32 t = 0; t < d.in.buffer_count; t++) {
                    float vol = m.in.mix_volumes[s][t] * m.in.volume;
                    if (vol == 0) continue;
                    float* src = buf(m.buffer_offset + s);
                    float* dst = buf(d.buffer_offset + t);
                    for (u32 k = 0; k < n; k++) dst[k] += src[k] * vol;
                }
        }
        run_effects(0);
        // sinks
        memset(out, 0, (size_t)n * 4);
        MixState& fin = mixes_[0];
        for (auto& sk : sinks_) {
            if (!sk.in_use) continue;
            if (sk.type == 1) {
                u32 ic = std::min<u32>(sk.device.input_count, kMaxChannels);
                float ch[kMaxChannels] = {};
                for (u32 k = 0; k < n; k++) {
                    for (u32 c = 0; c < ic; c++) {
                        s32 b = sk.device.inputs[c];
                        ch[c] = (b >= 0 && (u32)b < fin.in.buffer_count) ? buf(fin.buffer_offset + b)[k] * fin.in.volume : 0.0f;
                    }
                    float l, r;
                    if (ic >= 6) {
                        const float* dc = sk.device.downmix_enabled ? sk.device.downmix_coeff : nullptr;
                        float cf = dc ? dc[0] : 1.0f, cc = dc ? dc[1] : 0.707f, cl = dc ? dc[2] : 0.5f, cb = dc ? dc[3] : 0.707f;
                        l = ch[0] * cf + ch[2] * cc + ch[3] * cl + ch[4] * cb;
                        r = ch[1] * cf + ch[2] * cc + ch[3] * cl + ch[5] * cb;
                    } else {
                        l = ch[0];
                        r = ic >= 2 ? ch[1] : ch[0];
                    }
                    out[k * 2] = clamp16((s32)(out[k * 2] + l));
                    out[k * 2 + 1] = clamp16((s32)(out[k * 2 + 1] + r));
                }
            } else if (sk.type == 2 && sk.circular.cpu_address && sk.circular.size) {
                u32 ic = std::min<u32>(sk.circular.input_count, kMaxChannels);
                u32 frame_bytes = n * 2;  // per channel, s16
                u8* base = (u8*)(uintptr_t)sk.circular.cpu_address;
                for (u32 c = 0; c < ic; c++) {
                    s32 b = sk.circular.inputs[c];
                    for (u32 k = 0; k < n; k++) {
                        s16 v = (b >= 0 && (u32)b < fin.in.buffer_count) ? clamp16((s32)buf(fin.buffer_offset + b)[k]) : 0;
                        u32 off = (circ_pos_ + k * 2) % sk.circular.size;
                        memcpy(base + off, &v, 2);
                    }
                    circ_pos_ = (circ_pos_ + frame_bytes) % sk.circular.size;
                }
            }
        }
        frames_++;
    }

private:
    float* buf(u32 index) { return &mix_buffers_[(size_t)index * params_.sample_count]; }

    void update_voice(VoiceState& v, const VoiceIn& in) {
        if (!in.in_use) {
            v.active = false;
            v.in = in;
            return;
        }
        if (in.is_new || !v.active) {
            v = VoiceState{};
            v.active = true;
            v.index = in.wave_buffer_index % kMaxWaveBuffers;
            if (in.sample_format == 6 && in.src_data_address) memcpy(v.coeffs, (void*)(uintptr_t)in.src_data_address, 32);
        }
        bool was_stopped = v.in.play_state == 1;
        v.in = in;
        for (u32 i = 0; i < kMaxWaveBuffers; i++) {
            const WaveBufferIn& wb = in.wave[i];
            if (!wb.sent_to_dsp && wb.address && wb.size) {  // newly submitted
                v.slots[i].valid = true;
                v.slots[i].wb = wb;
            }
        }
        if (in.flush_buffer_count) {
            for (u32 i = 0; i < in.flush_buffer_count && i < kMaxWaveBuffers; i++) {
                if (v.slots[v.index].valid) v.consumed++;
                v.slots[v.index].valid = false;
                v.index = (v.index + 1) % kMaxWaveBuffers;
            }
            v.pos = 0;
            v.frac = 0;
        }
        if (in.play_state == 1 && !was_stopped) {  // Stopped: restart from the current buffer
            v.pos = 0;
            v.frac = 0;
            v.yn1 = v.yn2 = 0;
        }
    }

    // Next source sample for channel `ch` of voice v (interleaved PCM / mono ADPCM). Advances state
    // for the last channel only; returns false when nothing is queued.
    bool fetch(VoiceState& v, float* samples, u32 channels) {
        for (int guard = 0; guard < 8; guard++) {
            VoiceState::Slot& s = v.slots[v.index];
            if (!s.valid) return false;
            WaveBufferIn& wb = s.wb;
            s64 endpos = wb.end_offset;
            if (v.pos == 0 && wb.start_offset) v.pos = wb.start_offset;
            if (v.pos >= endpos || endpos <= wb.start_offset) {
                if (wb.loop && (v.in.sample_format != 6 || true) && (wb.loop_count < 0 || v.loops_left-- > 0)) {
                    v.pos = rev_ >= 5 && wb.loop_end > (u32)wb.start_offset ? wb.loop_start : wb.start_offset;
                    continue;
                }
                s.valid = false;
                v.consumed++;
                v.index = (v.index + 1) % kMaxWaveBuffers;
                v.pos = 0;
                v.loops_left = 0;
                VoiceState::Slot& ns = v.slots[v.index];
                if (ns.valid) {
                    v.loops_left = ns.wb.loop_count;
                    if (v.in.sample_format == 6 && ns.wb.context_address) {
                        const s16* ctx = (const s16*)(uintptr_t)ns.wb.context_address;
                        v.yn1 = ctx[1];
                        v.yn2 = ctx[2];
                    }
                }
                continue;
            }
            const u8* data = (const u8*)(uintptr_t)wb.address;
            switch (v.in.sample_format) {
            case 2: {  // PCM16 interleaved
                const s16* d = (const s16*)data;
                for (u32 c = 0; c < channels; c++) samples[c] = d[v.pos * channels + c];
                break;
            }
            case 1: {
                const s8* d = (const s8*)data;
                for (u32 c = 0; c < channels; c++) samples[c] = d[v.pos * channels + c] * 256.0f;
                break;
            }
            case 4: {
                const s32* d = (const s32*)data;
                for (u32 c = 0; c < channels; c++) samples[c] = d[v.pos * channels + c] / 65536.0f;
                break;
            }
            case 5: {
                const float* d = (const float*)data;
                for (u32 c = 0; c < channels; c++) samples[c] = d[v.pos * channels + c] * 32767.0f;
                break;
            }
            case 6: {  // Nintendo DSP-ADPCM, mono: 8-byte frames of 14 samples
                s64 frame = v.pos / 14;
                u32 nib = (u32)(v.pos % 14);
                const u8* f = data + frame * 8;
                if ((u64)(frame * 8 + 8) > wb.size) {
                    samples[0] = 0;
                    break;
                }
                u8 hdr = f[0];
                s32 scale = 1 << (hdr & 0xF);
                s32 c1 = v.coeffs[(hdr >> 4) * 2], c2 = v.coeffs[(hdr >> 4) * 2 + 1];
                u8 byte = f[1 + nib / 2];
                s32 n4 = (nib & 1) ? (byte & 0xF) : (byte >> 4);
                if (n4 >= 8) n4 -= 16;
                s32 sample = (((n4 * scale) << 11) + 1024 + c1 * v.yn1 + c2 * v.yn2) >> 11;
                sample = std::clamp(sample, -32768, 32767);
                v.yn2 = v.yn1;
                v.yn1 = (s16)sample;
                for (u32 c = 0; c < channels; c++) samples[c] = (float)sample;
                break;
            }
            default:
                for (u32 c = 0; c < channels; c++) samples[c] = 0;
                break;
            }
            v.pos++;
            v.played++;
            return true;
        }
        return false;
    }

    void render_voice(VoiceState& v) {
        if (!v.active || v.in.play_state != 0) return;  // Started only
        // A voice feeds either its mix directly (resource mix volumes) or, through a splitter, one
        // destination per channel (each with its own mix and volumes).
        bool split = v.in.splitter_id != 0xFFFFFFFFu && v.in.splitter_id < splitters_.size() && !splitters_[v.in.splitter_id].empty();
        if (!split && v.in.mix_id >= mixes_.size()) return;
        MixState& m = mixes_[split ? 0 : v.in.mix_id];
        if (!split && !m.in.in_use) return;
        u32 channels = std::clamp<u32>(v.in.channel_count, 1, kMaxChannels);
        u32 n = params_.sample_count;
        double ratio = (double)v.in.sample_rate * (v.in.pitch > 0 ? v.in.pitch : 1.0f) / params_.sample_rate;
        if (!(ratio > 0) || ratio > 32) ratio = 1;
        float tmp[kMaxChannels];
        for (u32 k = 0; k < n; k++) {
            v.frac += ratio;
            while (v.frac >= 1.0) {
                v.frac -= 1.0;
                if (!fetch(v, tmp, channels)) {
                    for (u32 c = 0; c < channels; c++) tmp[c] = 0;
                }
                for (u32 c = 0; c < channels; c++) {
                    v.prev[c][0] = v.prev[c][1];
                    v.prev[c][1] = tmp[c];
                }
            }
            float t = (float)v.frac;
            for (u32 c = 0; c < channels; c++) {
                float s = v.prev[c][0] + (v.prev[c][1] - v.prev[c][0]) * t;
                s *= v.in.volume;
                if (split) {
                    const auto& dests = splitters_[v.in.splitter_id];
                    s32 did = c < dests.size() ? dests[c] : -1;
                    if (did < 0 || (size_t)did >= splitter_dests_.size()) continue;
                    const SplitterDest& d = splitter_dests_[did];
                    if (!d.in_use || d.dest_mix_id < 0 || (size_t)d.dest_mix_id >= mixes_.size()) continue;
                    MixState& dm = mixes_[d.dest_mix_id];
                    if (!dm.in.in_use) continue;
                    for (u32 b = 0; b < dm.in.buffer_count && b < kMaxMixBuffers; b++) {
                        float vol = d.mix_volumes[b];
                        if (vol != 0) buf(dm.buffer_offset + b)[k] += s * vol;
                    }
                    continue;
                }
                u32 rid = v.in.channel_resource_ids[c];
                if (rid >= resources_.size()) continue;
                const ResourceIn& r = resources_[rid];
                for (u32 b = 0; b < m.in.buffer_count && b < kMaxMixBuffers; b++) {
                    float vol = r.mix_volumes[b];
                    if (vol != 0) buf(m.buffer_offset + b)[k] += s * vol;
                }
            }
        }
    }

    std::mutex m_;
    Params params_;
    u32 rev_;
    std::vector<VoiceState> voices_;
    std::vector<ResourceIn> resources_;
    struct SplitterDest {
        float mix_volumes[kMaxMixBuffers] = {};
        s32 dest_mix_id = -1;
        bool in_use = false;
    };
    std::vector<std::vector<s32>> splitters_;  // splitter id -> destination ids (one per voice channel)
    std::vector<SplitterDest> splitter_dests_;
    std::vector<MixState> mixes_;
    std::vector<SinkIn> sinks_;
    std::vector<u32> pools_;
    std::vector<EffectIn> effects_;
    std::vector<float> mix_buffers_;
    std::shared_ptr<ReadableEvent> event_;
    std::atomic<bool> started_{false};
    u64 frames_ = 0;
    u32 circ_pos_ = 0;
};

// All renderers tick on one audio thread.
static std::mutex g_renderers_lock;
static std::vector<std::shared_ptr<Renderer>> g_renderers;

static void audio_thread() {
    SetThreadDescription(GetCurrentThread(), L"HWDER audio render");
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    LARGE_INTEGER freq, next, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&next);
    std::vector<s16> frame, resampled;
    double out_frac = 0;
    s16 last[2] = {};
    for (;;) {
        std::vector<std::shared_ptr<Renderer>> rs;
        {
            std::lock_guard<std::mutex> l(g_renderers_lock);
            rs = g_renderers;
        }
        s64 period_ticks = freq.QuadPart / 200;  // 5 ms default
        for (auto& r : rs) {
            const Params& p = r->params();
            period_ticks = (s64)((double)p.sample_count / p.sample_rate * freq.QuadPart);
            if (!r->started()) continue;
            frame.resize((size_t)p.sample_count * 2);
            r->render(frame.data());
            if (p.sample_rate == kTargetRate) {
                g_ring.push(frame.data(), p.sample_count);
            } else {  // linear resample to 48 kHz
                double step = (double)p.sample_rate / kTargetRate;
                resampled.clear();
                for (u32 i = 0; i < p.sample_count; i++) {
                    while (out_frac < 1.0) {
                        float t = (float)out_frac;
                        resampled.push_back((s16)(last[0] + (frame[i * 2] - last[0]) * t));
                        resampled.push_back((s16)(last[1] + (frame[i * 2 + 1] - last[1]) * t));
                        out_frac += step;
                    }
                    out_frac -= 1.0;
                    last[0] = frame[i * 2];
                    last[1] = frame[i * 2 + 1];
                }
                g_ring.push(resampled.data(), (u32)resampled.size() / 2);
            }
        }
        for (auto& r : rs) r->event()->signal();
        next.QuadPart += period_ticks;
        QueryPerformanceCounter(&now);
        s64 wait = next.QuadPart - now.QuadPart;
        if (wait > 0) {
            LARGE_INTEGER due;
            due.QuadPart = -(wait * 10000000 / freq.QuadPart);
            SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0);
            WaitForSingleObject(timer, INFINITE);
        } else if (wait < -period_ticks * 8) {
            next = now;  // fell far behind: resync rather than burst
        }
    }
}

static void ensure_threads() {
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread(audio_thread).detach();
        std::thread(wasapi_thread).detach();
    });
}

static u64 work_buffer_size(const Params& p) {
    u64 size = 0x40 + (u64)p.mixes * 4 + (u64)p.sub_mixes * 256 * 4 + (u64)(p.sub_mixes + 1) * 0xA00;
    size += (u64)p.voices * 0x800;
    size += ((u64)(p.sinks + p.sub_mixes) * 240 * 4 + (u64)p.sample_count * 4) * (p.mixes + kMaxChannels) + 0x40;
    size += (u64)(p.effects + p.voices * kMaxWaveBuffers) * 0x40 + (u64)p.effects * 0x600 + (u64)p.sinks * 0x200;
    size += (u64)(p.splitter_infos + 1) * 0x100 + (u64)(p.splitter_destinations > 0 ? p.splitter_destinations : 0) * 0x100;
    size += (u64)p.perf_frames * 0x2000 + 0x18000 + 0x10000;
    return (size + 0xFFF) & ~0xFFFull;
}

// ------------------------------------------------------------------ services
class IAudioRenderer : public SimpleService {
public:
    explicit IAudioRenderer(std::shared_ptr<Renderer> r) : SimpleService("IAudioRenderer"), r_(std::move(r)) {
        reg(0, [this](Request&, Response& rs) { rs.push<u32>(r_->params().sample_rate); });
        reg(1, [this](Request&, Response& rs) { rs.push<u32>(r_->params().sample_count); });
        reg(2, [this](Request&, Response& rs) { rs.push<u32>(r_->params().mixes); });
        reg(3, [this](Request&, Response& rs) { rs.push<u32>(r_->started() ? 0 : 1); });
        auto update = [this](Request& rq, Response& rs) {
            Buffer in = rq.in_buffer(0), out = rq.out_buffer(0), perf = rq.out_buffer(1);
            if (perf.addr && perf.size) memset((void*)perf.addr, 0, perf.size);
            rs.result = r_->update((const u8*)in.addr, in.size, (u8*)out.addr, out.size);
        };
        reg(4, update);
        reg(10, update);
        reg(5, [this](Request&, Response&) { r_->start(); });
        reg(6, [this](Request&, Response&) { r_->stop(); });
        reg(7, [this](Request&, Response& rs) { push_handle(rs, r_->event()); });
        reg(8, [this](Request& rq, Response&) { limit_ = rq.pop<u32>(); });
        reg(9, [this](Request&, Response& rs) { rs.push<u32>(limit_); });
        nop({11});
    }
    std::shared_ptr<Renderer> r_;
    u32 limit_ = 100;
};

class IAudioDevice : public SimpleService {
public:
    IAudioDevice() : SimpleService("IAudioDevice") {
        auto list = [](Request& rq, Response& rs) {
            static const char* names[] = {"AudioTvOutput", "AudioStereoJackOutput", "AudioBuiltInSpeakerOutput"};
            Buffer b = rq.out_buffer(0);
            u32 n = (u32)std::min<u64>(3, b.size / 0x100);
            for (u32 i = 0; i < n; i++) {
                char nm[0x100] = {};
                strcpy(nm, names[i]);
                memcpy((u8*)b.addr + i * 0x100, nm, 0x100);
            }
            rs.push<s32>((s32)n);
        };
        reg(0, list);
        reg(6, list);
        nop({1, 7});
        ret<float>(2, 1.0f);
        ret<float>(8, 1.0f);
        auto active = [](Request& rq, Response&) {
            char nm[0x100] = "AudioTvOutput";
            write_buffer(rq.out_buffer(0), nm, sizeof(nm));
        };
        reg(3, active);
        reg(10, active);
        reg(4, [](Request&, Response& rs) { push_handle(rs, make_event()); });
        ret<u32>(5, 2);
        reg(11, [](Request&, Response& rs) { push_handle(rs, make_event()); });
        reg(12, [](Request&, Response& rs) { push_handle(rs, make_event()); });
        reg(13, [](Request&, Response& rs) { push_handle(rs, make_event()); });
    }
};

class AudRenU : public SimpleService {
public:
    AudRenU() : SimpleService("audren:u") {
        reg(0, [](Request& rq, Response& rs) {  // OpenAudioRenderer(params, tmem size, aruid) + handles
            Params p = rq.pop<Params>();
            ensure_threads();
            auto r = std::make_shared<Renderer>(p);
            {
                std::lock_guard<std::mutex> l(g_renderers_lock);
                g_renderers.push_back(r);
            }
            rs.push_object(std::make_shared<IAudioRenderer>(r));
        });
        reg(1, [](Request& rq, Response& rs) {  // GetWorkBufferSize
            Params p = rq.pop<Params>();
            rs.push<u64>(work_buffer_size(p));
        });
        obj(2, [] { return std::make_shared<IAudioDevice>(); });
        obj(4, [] { return std::make_shared<IAudioDevice>(); });
    }
};

// audout:u: direct PCM output sessions (not used by this game's engine, but kept functional).
class IAudioOut : public SimpleService {
public:
    IAudioOut() : SimpleService("IAudioOut") {
        ret<u32>(0, 1);  // GetAudioOutState: stopped
        nop({1, 2, 3});
        reg(4, [](Request& rq, Response& rs) {  // GetReleasedAudioOutBuffer -> count
            rs.push<u32>(0);
        });
        ret<u8>(5, 0);
        reg(6, [](Request&, Response& rs) { push_handle(rs, make_event(true)); });  // RegisterBufferEvent
        ret<u32>(7, 0);
    }
};
class AudOutU : public SimpleService {
public:
    AudOutU() : SimpleService("audout:u") {
        reg(0, [](Request&, Response& rs) { rs.push<u32>(0); });  // ListAudioOuts
        reg(1, [](Request& rq, Response& rs) {  // OpenAudioOut
            rs.push<u32>(kTargetRate);
            rs.push<u32>(2);
            rs.push<u32>(2);  // PcmInt16
            rs.push<u32>(1);  // state stopped
            rs.push_object(std::make_shared<IAudioOut>());
        });
        reg(2, [](Request&, Response& rs) { rs.push<u32>(0); });
    }
};

}  // namespace

void register_audio_services() {
    register_as<AudRenU>({"audren:u"});
    register_as<AudOutU>({"audout:u"});
}

}  // namespace ipc
