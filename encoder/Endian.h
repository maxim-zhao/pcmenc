#pragma once
#include <cstdint>

// constexpr magic to determine endianness
class Endian
{
    static constexpr uint32_t uint32_ = 0x01020304;
#ifdef MSVC_VER
#pragma warning(disable:4310) // Truncating constant
#endif
    static constexpr uint8_t magic_ = (const uint8_t&)uint32_;
public:
    static constexpr bool little = magic_ == 0x04;
    static constexpr bool middle = magic_ == 0x02;
    static constexpr bool big = magic_ == 0x01;
    static_assert(little || middle || big, "Cannot determine endianness!");
private:
    Endian() = delete;
};
