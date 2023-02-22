#pragma once
#include <cstdint>

constexpr auto StringToMarker(const char s[5])
{
    return
        (static_cast<uint32_t>(s[0]) << 0) |
        (static_cast<uint32_t>(s[1]) << 8) |
        (static_cast<uint32_t>(s[2]) << 16) |
        (static_cast<uint32_t>(s[3]) << 24);
}
