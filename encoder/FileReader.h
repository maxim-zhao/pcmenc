#pragma once
#include <fstream>


// Helper class for file IO
class FileReader
{
private:
    std::ifstream _f;
    size_t _size;

public:
    explicit FileReader(const std::string& filename);

    ~FileReader();
    FileReader(const FileReader& other) = delete;
    FileReader(FileReader&& other) noexcept = default;
    FileReader& operator=(const FileReader& other) = delete;
    FileReader& operator=(FileReader&& other) noexcept = default;

    uint32_t read32();

    uint16_t read16();

    uint8_t read();

    FileReader& checkMarker(const char marker[5]);

    [[nodiscard]]
    size_t size() const;

    FileReader& seek(uint32_t offset);
};

