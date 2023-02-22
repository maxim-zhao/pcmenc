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
    return static_cast<uint32_t>(
        (read() << 0) |
        (read() << 8) |
        (read() << 16) |
        (read() << 24));
}

uint16_t FileReader::read16()
{
    return static_cast<uint16_t>(
        (read() << 0) |
        (read() << 8));
}

uint8_t FileReader::read()
{
    uint8_t v;
    _f.read(reinterpret_cast<char*>(&v), sizeof(uint8_t));
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

