// Small system services: lm, apm, acc, set, time, pctl, prepo, friend, nifm, aoc, ns, ...
#include <ctime>

#include "common.h"

extern "C" u64 hw_cntvct();

namespace ipc {

namespace {

// ------------------------------------------------------------------ lm (logging)
class ILogger : public SimpleService {
public:
    ILogger() : SimpleService("ILogger") {
        reg(0, [](Request& rq, Response&) {  // Log(X buffer: LogPacket)
            Buffer b = rq.in_buffer();
            if (!b.addr || b.size < 0x10) return;
            // Packet header 0x10 bytes, then TLV fields (u8 type, uleb128 size, data). Type 2 = message.
            const u8* p = (const u8*)b.addr + 0x10;
            const u8* end = (const u8*)b.addr + b.size;
            while (p < end) {
                u8 type = *p++;
                u32 size = 0, shift = 0;
                while (p < end) {
                    u8 c = *p++;
                    size |= (u32)(c & 0x7F) << shift;
                    shift += 7;
                    if (!(c & 0x80)) break;
                }
                if (p + size > end) break;
                if (type == 2 && size) hw_log("guest: %.*s", (int)size, (const char*)p);
                p += size;
            }
        });
        nop({1});
    }
};
class Lm : public SimpleService {
public:
    Lm() : SimpleService("lm") { obj(0, [] { return std::make_shared<ILogger>(); }); }
};

// ------------------------------------------------------------------ apm (performance)
class ApmSession : public SimpleService {
public:
    ApmSession() : SimpleService("apm:ISession") {
        nop({0});
        ret<u32>(1, 0x00020003);  // GetPerformanceConfiguration
        nop({2});
    }
};
class Apm : public SimpleService {
public:
    Apm() : SimpleService("apm") {
        obj(0, [] { return std::make_shared<ApmSession>(); });
        ret<u32>(1, 1);  // GetPerformanceMode: boost
        ret<u8>(6, 0);   // IsEnoughPowerSupplied
    }
};

// ------------------------------------------------------------------ acc (accounts)
struct Uid {
    u64 lo, hi;
};
constexpr Uid kUser{0x0000000000000001ull, 0x48574445525F5553ull};
const char* kNickname = "Link";

class IProfile : public SimpleService {
public:
    IProfile() : SimpleService("acc:IProfile") {
        auto base = [](Response& rs) {
            u8 pb[0x38] = {};
            memcpy(pb, &kUser, 16);
            u64 ts = (u64)time(nullptr);
            memcpy(pb + 0x10, &ts, 8);
            strncpy((char*)pb + 0x18, kNickname, 0x20);
            rs.data.insert(rs.data.end(), pb, pb + sizeof(pb));
        };
        reg(0, [base](Request& rq, Response& rs) {  // Get: UserData to B buffer + ProfileBase
            u8 ud[0x80] = {};
            write_buffer(rq.out_buffer(), ud, sizeof(ud));
            base(rs);
        });
        reg(1, [base](Request&, Response& rs) { base(rs); });
        // A tiny valid JPEG is not required by most games; report a 0-byte image.
        ret<u32>(10, 0);
        reg(11, [](Request&, Response& rs) { rs.push<u32>(0); });
    }
};
class IManagerForApplication : public SimpleService {
public:
    IManagerForApplication() : SimpleService("acc:IManagerForApplication") {
        ret<u8>(0, 0);                         // CheckAvailability
        ret<u64>(1, 0x48574445525F0001ull);    // GetAccountId
        reg(2, [](Request&, Response& rs) { rs.result = ModuleResult(124, 3); });  // EnsureIdTokenCacheAsync: no network
    }
};
class Acc : public SimpleService {
public:
    Acc() : SimpleService("acc:u0") {
        ret<s32>(0, 1);  // GetUserCount
        ret<u8>(1, 1);   // GetUserExistence
        auto list = [](Request& rq, Response&) {
            Uid ids[8] = {kUser};
            write_buffer(rq.out_buffer(), ids, sizeof(ids));
        };
        reg(2, list);  // ListAllUsers
        reg(3, list);  // ListOpenUsers
        reg(4, [](Request&, Response& rs) { rs.push(kUser); });  // GetLastOpenedUser
        obj(5, [] { return std::make_shared<IProfile>(); });       // GetProfile
        ret<u8>(50, 0);                                            // IsUserRegistrationRequestPermitted
        reg(51, [](Request&, Response& rs) { rs.push(kUser); });  // TrySelectUserWithoutInteraction
        nop({100, 102, 103, 110, 111, 140, 150});
        obj(101, [] { return std::make_shared<IManagerForApplication>(); });
        ret<u8>(160, 0);
    }
};

// ------------------------------------------------------------------ set (settings)
u64 lang_code(const char* s) {
    u64 v = 0;
    memcpy(&v, s, strlen(s));
    return v;
}
const char* kLanguages[] = {"ja", "en-US", "fr", "de", "it", "es", "zh-CN", "ko", "nl", "pt", "ru", "zh-TW",
                            "en-GB", "fr-CA", "es-419"};
constexpr int kLanguageCount = 15;

class Set : public SimpleService {
public:
    Set() : SimpleService("set") {
        reg(0, [](Request&, Response& rs) { rs.push(lang_code("en-US")); });  // GetLanguageCode
        auto langs = [](Request& rq, Response& rs) {
            u64 codes[kLanguageCount];
            for (int i = 0; i < kLanguageCount; i++) codes[i] = lang_code(kLanguages[i]);
            Buffer b = rq.out_buffer();
            u32 n = (u32)std::min<u64>(kLanguageCount, b.size / 8);
            write_buffer(b, codes, n * 8);
            rs.push<s32>((s32)n);
        };
        reg(1, langs);
        reg(5, langs);
        ret<s32>(2, 1);   // MakeLanguageCode
        ret<s32>(3, kLanguageCount);
        ret<s32>(6, kLanguageCount);
        ret<s32>(4, 1);   // GetRegionCode: USA
        ret<u8>(8, 0);    // GetQuestFlag
        reg(9, [](Request& rq, Response&) {  // GetKeyCodeMap
            Buffer b = rq.out_buffer();
            if (b.addr) memset((void*)b.addr, 0, b.size);
        });
        reg(10, [](Request& rq, Response& rs) {  // GetDeviceNickName
            char n[0x80] = "HWDER";
            write_buffer(rq.out_buffer(), n, sizeof(n));
        });
    }
};
class SetSys : public SimpleService {
public:
    SetSys() : SimpleService("set:sys") {
        auto fw = [](Request& rq, Response&) {  // GetFirmwareVersion(2): 0x100-byte struct
            u8 v[0x100] = {};
            v[0] = 3; v[1] = 0; v[2] = 1; v[4] = 1;
            strcpy((char*)v + 0x8, "NintendoSDK Firmware for NX");
            strcpy((char*)v + 0x28, "3.0.1-1.0");
            strcpy((char*)v + 0x68, "3.0.1");
            strcpy((char*)v + 0x80, "NintendoSDK Firmware for NX 3.0.1-1.0");
            write_buffer(rq.out_buffer(), v, sizeof(v));
        };
        reg(3, fw);
        reg(4, fw);
        ret<s32>(23, 0);  // GetColorSetId
        reg(37, [](Request&, Response& rs) { rs.push<u64>(8); });  // GetSettingsItemValueSize
        reg(38, [](Request& rq, Response& rs) {                    // GetSettingsItemValue(class, key)
            std::string cls = rq.in_string(0), key = rq.in_string(1);
            hw_log("set:sys: GetSettingsItemValue(%s, %s)", cls.c_str(), key.c_str());
            u64 zero = 0;
            write_buffer(rq.out_buffer(), &zero, 8);
            rs.push<u64>(8);
        });
        ret<u8>(62, 0);  // GetDebugModeFlag
        ret<u8>(77, 0);
    }
};

// ------------------------------------------------------------------ time
// Steady clock: seconds since boot from CNTVCT (19.2MHz). System clock: host UTC.
const u64 kClockSource[2] = {0x4857444552434C4Bull, 0x0000000000000001ull};
s64 steady_seconds() { return (s64)(hw_cntvct() / 19200000ull); }

class ISystemClock : public SimpleService {
public:
    ISystemClock() : SimpleService("time:ISystemClock") {
        reg(0, [](Request&, Response& rs) { rs.push<s64>((s64)time(nullptr)); });  // GetCurrentTime
        nop({1});
        reg(2, [](Request&, Response& rs) {  // GetSystemClockContext
            s64 steady = steady_seconds();
            rs.push<s64>((s64)time(nullptr) - steady);
            rs.push<s64>(steady);
            rs.push(kClockSource[0]);
            rs.push(kClockSource[1]);
        });
        nop({3});
    }
};
class ISteadyClock : public SimpleService {
public:
    ISteadyClock() : SimpleService("time:ISteadyClock") {
        reg(0, [](Request&, Response& rs) {  // GetCurrentTimePoint
            rs.push<s64>(steady_seconds());
            rs.push(kClockSource[0]);
            rs.push(kClockSource[1]);
        });
        ret<s64>(100, 0);  // GetTestOffset
        nop({101});
    }
};

// nn::time::CalendarTime (8 bytes) + CalendarAdditionalInfo (0x18 bytes)
void push_calendar(Response& rs, s64 t, bool local) {
    time_t tt = (time_t)t;
    struct tm tmv;
    if (local) localtime_s(&tmv, &tt);
    else gmtime_s(&tmv, &tt);
    rs.push<u16>((u16)(tmv.tm_year + 1900));
    rs.push<u8>((u8)(tmv.tm_mon + 1));
    rs.push<u8>((u8)tmv.tm_mday);
    rs.push<u8>((u8)tmv.tm_hour);
    rs.push<u8>((u8)tmv.tm_min);
    rs.push<u8>((u8)tmv.tm_sec);
    rs.push<u8>(0);
    rs.push<u32>((u32)tmv.tm_wday);
    rs.push<u32>((u32)tmv.tm_yday);
    char tz[8] = "UTC";
    rs.data.insert(rs.data.end(), tz, tz + 8);
    rs.push<u32>(tmv.tm_isdst > 0);
    long off = 0;
    if (local) _get_timezone(&off);
    rs.push<s32>(local ? (s32)-off : 0);
}
s64 from_calendar(Request& rq) {
    struct tm tmv = {};
    tmv.tm_year = rq.pop<u16>() - 1900;
    tmv.tm_mon = rq.pop<u8>() - 1;
    tmv.tm_mday = rq.pop<u8>();
    tmv.tm_hour = rq.pop<u8>();
    tmv.tm_min = rq.pop<u8>();
    tmv.tm_sec = rq.pop<u8>();
    return (s64)_mkgmtime(&tmv);
}

class ITimeZoneService : public SimpleService {
public:
    ITimeZoneService() : SimpleService("time:ITimeZoneService") {
        reg(0, [](Request&, Response& rs) {  // GetDeviceLocationName
            char n[0x24] = "UTC";
            rs.data.insert(rs.data.end(), n, n + sizeof(n));
        });
        nop({1});
        ret<u32>(2, 1);  // GetTotalLocationNameCount
        reg(3, [](Request& rq, Response& rs) {  // LoadLocationNameList
            char n[0x24] = "UTC";
            write_buffer(rq.out_buffer(), n, sizeof(n));
            rs.push<u32>(1);
        });
        nop({4});  // LoadTimeZoneRule: our rule buffer is opaque, conversions below ignore it
        reg(5, [](Request&, Response& rs) {  // GetTimeZoneRuleVersion
            char v[0x10] = "2018c";
            rs.data.insert(rs.data.end(), v, v + 16);
        });
        reg(100, [](Request& rq, Response& rs) { push_calendar(rs, rq.pop<s64>(), false); });  // ToCalendarTime
        reg(101, [](Request& rq, Response& rs) { push_calendar(rs, rq.pop<s64>(), false); });  // ...WithMyRule
        auto to_posix = [](Request& rq, Response& rs) {
            s64 t = from_calendar(rq);
            write_buffer(rq.out_buffer(), &t, 8);
            rs.push<u32>(1);
        };
        reg(201, to_posix);
        reg(202, to_posix);
    }
};

class Time : public SimpleService {
public:
    explicit Time(const char* name = "time") : SimpleService(name) {
        obj(0, [] { return std::make_shared<ISystemClock>(); });  // StandardUserSystemClock
        obj(1, [] { return std::make_shared<ISystemClock>(); });  // StandardNetworkSystemClock
        obj(2, [] { return std::make_shared<ISteadyClock>(); });
        obj(3, [] { return std::make_shared<ITimeZoneService>(); });
        obj(4, [] { return std::make_shared<ISystemClock>(); });  // StandardLocalSystemClock
        obj(5, [] { return std::make_shared<ISystemClock>(); });
        ret<u8>(100, 0);  // IsStandardUserSystemClockAutomaticCorrectionEnabled
        nop({101});
    }
};

// ------------------------------------------------------------------ pctl (parental controls)
class IParentalControlService : public SimpleService {
public:
    IParentalControlService() : SimpleService("pctl:IParentalControlService") {
        nop({1, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1013, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039});
        ret<u8>(1031, 0);  // IsRestrictionEnabled
        ret<u8>(1006, 0);  // IsRestrictionTemporaryUnlocked
        ret<u32>(1035, 0); // GetSafetyLevel
        ret<u8>(1033, 0);
    }
};
class Pctl : public SimpleService {
public:
    Pctl() : SimpleService("pctl") {
        obj(0, [] { return std::make_shared<IParentalControlService>(); });
        obj(1, [] { return std::make_shared<IParentalControlService>(); });
    }
};

// ------------------------------------------------------------------ prepo / friend / nifm / ns / aoc / ...
class Prepo : public SimpleService {
public:
    Prepo() : SimpleService("prepo") {}  // all reports are accepted and dropped
};

class IFriendService : public SimpleService {
public:
    IFriendService() : SimpleService("friend:IFriendService") {
        reg(0, [](Request&, Response& rs) { push_handle(rs, make_event()); });  // GetCompletionEvent
        ret<s32>(10101, 0);  // GetFriendList
        ret<s32>(10100, 0);
    }
};
class Friend : public SimpleService {
public:
    Friend() : SimpleService("friend") {
        obj(0, [] { return std::make_shared<IFriendService>(); });
        obj(1, [] {
            auto s = std::make_shared<SimpleService>("friend:INotificationService");
            return s;
        });
    }
};

class IRequest : public SimpleService {
public:
    IRequest() : SimpleService("nifm:IRequest") {
        ret<u32>(0, 1);  // GetRequestState: Invalid/not connected
        reg(1, [](Request&, Response& rs) { rs.result = ModuleResult(110, 300); });  // GetResult: no connection
        reg(2, [](Request&, Response& rs) {  // GetSystemEventReadableHandles
            push_handle(rs, make_event(true));
            push_handle(rs, make_event(true));
        });
        nop({3, 4, 6, 8, 11, 21});
    }
};
class IGeneralService : public SimpleService {
public:
    IGeneralService() : SimpleService("nifm:IGeneralService") {
        reg(1, [](Request&, Response& rs) {  // GetClientId
            rs.push<u32>(1);
        });
        obj(4, [] { return std::make_shared<IRequest>(); });  // CreateRequest
        reg(5, [](Request&, Response& rs) { rs.result = ModuleResult(110, 300); });  // GetCurrentNetworkProfile
        reg(12, [](Request&, Response& rs) { rs.result = ModuleResult(110, 300); });  // GetCurrentIpAddress
        ret<u8>(18, 0);  // GetInternetConnectionStatus
        ret<u8>(21, 0);  // IsAnyInternetRequestAccepted
        ret<u8>(22, 0);  // IsAnyForegroundRequestAccepted
    }
};
class Nifm : public SimpleService {
public:
    Nifm() : SimpleService("nifm") {
        obj(4, [] { return std::make_shared<IGeneralService>(); });  // CreateGeneralServiceOld
        obj(5, [] { return std::make_shared<IGeneralService>(); });  // CreateGeneralService
    }
};

class Aoc : public SimpleService {
public:
    Aoc() : SimpleService("aoc:u") {
        // The base game dump has no add-on content installed.
        ret<u32>(0, 0);  // CountAddOnContentByApplicationId
        ret<u32>(1, 0);  // ListAddOnContentByApplicationId
        ret<u32>(2, 0);  // CountAddOnContent
        ret<u32>(3, 0);  // ListAddOnContent
        ret<u64>(4, 0x0100AE00096EB000ull);  // GetAddOnContentBaseIdByApplicationId
        ret<u64>(5, 0x0100AE00096EB000ull);  // GetAddOnContentBaseId
        nop({6, 7, 8});
        reg(8, [](Request&, Response& rs) { push_handle(rs, make_event()); });  // GetAddOnContentListChangedEvent
    }
};

class Ns : public SimpleService {
public:
    Ns() : SimpleService("ns") {}
};

class Ssl : public SimpleService {
public:
    Ssl() : SimpleService("ssl") {}
};

}  // namespace

void register_misc_services() {
    register_as<Lm>({"lm"});
    register_as<Apm>({"apm", "apm:p", "apm:sys"});
    register_as<Acc>({"acc:u0", "acc:u1", "acc:su", "acc:aa"});
    register_as<Set>({"set"});
    register_as<SetSys>({"set:sys", "set:cal", "set:fd"});
    register_service("time:u", [] { return std::make_shared<Time>("time:u"); });
    register_service("time:a", [] { return std::make_shared<Time>("time:a"); });
    register_service("time:s", [] { return std::make_shared<Time>("time:s"); });
    register_as<Pctl>({"pctl", "pctl:a", "pctl:s", "pctl:r"});
    register_as<Prepo>({"prepo:u", "prepo:a", "prepo:s", "prepo:m"});
    register_as<Friend>({"friend:u", "friend:a", "friend:m", "friend:s", "friend:v"});
    register_as<Nifm>({"nifm:u", "nifm:a", "nifm:s"});
    register_as<Aoc>({"aoc:u"});
    register_as<Ns>({"ns:am", "ns:am2", "ns:ec", "ns:rid", "ns:rt", "ns:su", "ns:vm", "ns:web"});
    register_as<Ssl>({"ssl"});
}

}  // namespace ipc
