#pragma once
#include <string>
#include <map>

// A class which parses the commandline args into an internal dictionary, and then does some type conversion for us.
class Args
{
    std::map<std::string, std::string> _args;

public:
    Args(int argc, char** argv);

    [[nodiscard]] std::string getString(const std::string& name, const std::string& defaultValue);

    [[nodiscard]] int getInt(const std::string& name, int defaultValue);

    [[nodiscard]] bool exists(const std::string& name) const;
};
