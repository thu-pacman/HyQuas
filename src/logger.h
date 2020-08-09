#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <stdarg.h>

class Logger {
    static Logger* instance;
public:
    static void add(const char* format, ...) {
#ifdef SHOW_SUMMARY
        Logger::init();
        char buffer[1024];
        va_list args;
        va_start(args, format);
        vsprintf(buffer, format, args);
        va_end(args);
        instance -> infos.push_back(std::string(buffer));
#endif
    }

    inline static void print() {
#ifdef SHOW_SUMMARY
        Logger::init();
        for (auto& s: instance -> infos) {
            std::cout << "Logger: " << s << std::endl;
        }
        instance -> infos.clear();
#endif
    }

private:
    Logger() = default;
    static void init() {
        if (instance == NULL) {
            instance = new Logger();
        }
    }
private:
    std::vector<std::string> infos;
};
