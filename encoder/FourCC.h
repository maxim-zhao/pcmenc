#pragma once
#include <cstdint>

constexpr auto StringToMarker(const char s[5])
{
    return (uint32_t)s[0] << 0 | (uint32_t)s[1] << 8 | (uint32_t)s[2] << 16 | (uint32_t)s[3] << 24;
}
