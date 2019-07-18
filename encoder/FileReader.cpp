#include "FileReader.h"
#include "Endian.h"
#include "FourCC.h"
#include <sstream>

FileReader::FileReader(const std::string& filename)
{
    _f.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
    _f.open(filename, std::ifstream::binary);
    _f.seekg(0, std::ios::end);
    _size = static_cast<size_t>(_f.tellg());
    _f.seekg(0, std::ios::beg);
}

FileReader::~FileReader()
{
    try
    {
        _f.close();
    }
    catch (std::exception&)
    {
        // do nothing
    }
}

uint32_t FileReader::read32()
{
    uint32_t v;
    _f.read((char*)&v, sizeof(uint32_t));
    // ReSharper disable once CppUnreachableCode
    if constexpr (Endian::big)
    {
        v = (v >> 24 & 0x000000ff) | (v >> 8 & 0x0000ff00) |
            (v << 8 & 0x00ff0000) | (v << 24 & 0xff000000);
    }
    return v;
} 

uint16_t FileReader::read16()
{
    uint16_t v;
    _f.read((char*)&v, sizeof(uint16_t));
    // ReSharper disable once CppUnreachableCode
    if constexpr (Endian::big)
    {
        v = (v << 8 & 0x00ff0000) | (v << 24 & 0xff000000);
    }
    return v;
}

uint8_t FileReader::read()
{
    uint8_t v;
    _f.read((char*)&v, sizeof(uint8_t));
    return v;
}

FileReader& FileReader::checkMarker(const char marker[5])
{
    if (read32() != StringToMarker(marker))
    {
        std::ostringstream ss;
        ss << "Marker " << marker << " not found in file";
        throw std::runtime_error(ss.str());
    }
    return *this;
}

size_t FileReader::size() const
{
    return _size;
}

FileReader& FileReader::seek(const uint32_t offset)
{
    _f.seekg(offset, std::ios::cur);
    return *this;
}

