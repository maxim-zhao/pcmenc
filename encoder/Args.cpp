#include "Args.h"

Args::Args(int argc, char** argv)
{
    bool haveLastKey = false;
    std::map<std::string, std::string>::iterator lastKey;
    for (int i = 1; i < argc; ++i)
    {
        switch (argv[i][0])
        {
        case '/':
        case '-':
            // Store as a valueless key
            lastKey = _args.insert(make_pair(std::string(argv[i] + 1), "")).first;
            haveLastKey = true;
            // Remember it
            break;
        case '\0':
            break;
        default:
            // Must be a value for the last key, or a filename
            if (haveLastKey)
            {
                lastKey->second = argv[i];
            }
            else
            {
                _args.insert(std::make_pair("filename", argv[i]));
            }
            // Clear it so we don't put the filename in the wrong place
            haveLastKey = false;
            break;
        }
    }
}

std::string Args::getString(const std::string& name, const std::string& defaultValue)
{
    const auto it = _args.find(name);
    if (it == _args.end())
    {
        return defaultValue;
    }
    return it->second;
}

int Args::getInt(const std::string& name, uint32_t defaultValue)
{
    const auto it = _args.find(name);
    if (it == _args.end())
    {
        return defaultValue;
    }
    return std::strtol(it->second.c_str(), nullptr, 10);
}

bool Args::exists(const std::string& name) const
{
    return _args.find(name) != _args.end();
}
