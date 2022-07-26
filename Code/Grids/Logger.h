#pragma once

#include <stdio.h>
#include <string>
#include <iostream>
#include <cstdarg>
#include <fstream>
#include <vector>

#ifndef _TRACER_H_
#define _TRACER_H_

enum class EVerboseLevel
{
    CRUCIAL,
    GENERAL,
    DETAILED,
    PARANOIAC,
};


const int _kTraceBuffSize = 32768;

__forceinline void GetTimeNow(char* outchar, size_t buffSize)
{
    time_t now = time(0);
#if WIN64
    //ctime_s(outchar, buffSize, &now);
    tm now_tm;
    //gmtime_s(&now_tm, &now);
    localtime_s(&now_tm, &now);
    strftime(outchar, buffSize, "%d-%m-%Y %H-%M-%S", &now_tm);
#else
    tm now_tm = *localtime(&now);
    strftime(outchar, buffSize, "%d-%m-%Y %H-%M-%S", &now_tm);
#endif
}

#if !WIN64

inline int sprintf_s(char* buffer, size_t sizeOfBuffer, const char* format, ...)
{
    va_list ap;
    va_start(ap, format);
    int result = std::vsnprintf(buffer, sizeOfBuffer, format, ap);
    va_end(ap);
    return result;
}

inline int vsprintf_s(char* buffer, size_t sizeOfBuffer, const char* format, va_list ap)
{
    return std::vsnprintf(buffer, sizeOfBuffer, format, ap);
}

inline int vsnprintf_s(char* buffer, size_t sizeOfBuffer, const char* format, va_list ap)
{
    return std::vsnprintf(buffer, sizeOfBuffer, format, ap);
}

inline char* strcpy_s(char* dest, size_t length, const char* source)
{
    return std::strncpy(dest, source, length);
}

#endif

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    if (size_s <= 0) 
    { 
        return "Error during formatting.";
    }
    size_t size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1);
}

class Logger
{
public:
    Logger(void)
        : m_eLevel(EVerboseLevel::PARANOIAC)
        , m_pStream(nullptr)
        , m_pStdStream(nullptr)
        , m_bLogDate(true)
    {
        m_pStdStream = new std::ostream(std::cout.rdbuf());
        std::string sRealFile;
        static char datetime[256];
        GetTimeNow(datetime, 256);
        char filename[256];
        sprintf_s(filename, "%s.log", datetime);
        m_pStream = new std::ofstream(filename);

        if (nullptr == m_pStdStream || nullptr == m_pStream)
        {
            printf("ERROR: Logger: no output stream!!");
            if (NULL != m_pStream)
            {
                m_pStream->flush();
            }
            exit(EXIT_FAILURE);
        }
        memset(m_cBuff, 0, sizeof(char) * _kTraceBuffSize);
    }

    ~Logger(void)
    {
        if (nullptr != m_pStream)
        {
            m_pStream->flush();
            delete m_pStream;
            m_pStream = nullptr;
        }
    }

    inline void SetVerboseLevel(EVerboseLevel eLevel) { m_eLevel = eLevel; }

    inline void Print(EVerboseLevel level, const char* format, va_list& arg)
    {
        if ((level <= m_eLevel))
        {
            if (EVerboseLevel::CRUCIAL == level)
            {
                *m_pStdStream << "\033[31;1m";
            }
            if (m_bLogDate)
            {
                static char timeBuffer[256];
                if (level <= EVerboseLevel::GENERAL)
                {
                    GetTimeNow(timeBuffer, 256);
                    *m_pStdStream << "[" << timeBuffer << "|" << "]";
                    if (nullptr != m_pStream)
                    {
                        *m_pStream << "[" << timeBuffer << "|" << "]";
                    }
                }
            }
            vsnprintf_s(m_cBuff, _kTraceBuffSize - 1, format, arg);
            *m_pStdStream << m_cBuff;
            if (EVerboseLevel::CRUCIAL == level)
            {
                *m_pStdStream << "\033[0m";
            }
            if (nullptr != m_pStream)
            {
                *m_pStream << m_cBuff;
#ifdef DEBUG
                * m_pStream << std::flush;
#endif
            }
        }
    }

    inline void Flush() const
    {
        if (nullptr != m_pStream)
        {
            m_pStream->flush();
        }
    }

    inline void SetLogDate(bool bLog) 
    { 
        m_bLogDate = bLog; 
    }

    inline void PushLogDate(bool bNew) 
    {
        m_lstLogDate.push_back(m_bLogDate);
        m_bLogDate = bNew;
    }

    inline void PopLogDate()
    {
        if (m_lstLogDate.size() > 0)
        {
            m_bLogDate = m_lstLogDate[m_lstLogDate.size() - 1];
            m_lstLogDate.pop_back();
        }
    }

    inline bool GetLogDate() const { return m_bLogDate; }

private:

    EVerboseLevel m_eLevel;
    std::ostream* m_pStream;
    std::ostream* m_pStdStream;
    char m_cBuff[_kTraceBuffSize];
    bool m_bLogDate;
    std::vector<bool> m_lstLogDate;
};

extern void LogOut(EVerboseLevel eLevel, const char* format, ...);
extern void _LogCrucial(const char* format, ...);
extern void LogGeneral(const char* format, ...);
extern void LogDetailed(const char* format, ...);
extern void LogParanoiac(const char* format, ...);

#ifdef DEBUG
#   define LogCrucial(...) {char ___msg[1024];sprintf_s(___msg, 1024, __VA_ARGS__);_LogCrucial("%s(%d): Error: %s\n", __FILE__, __LINE__, ___msg);}
#else
#   define LogCrucial(...) {_LogCrucial(__VA_ARGS__);}
#endif

extern Logger GLogger;

class CLogger
{
public:
    CLogger() 
        : m_eVerb(EVerboseLevel::PARANOIAC)
    {
    }
    void SetVerb(EVerboseLevel eVerb) { m_eVerb = eVerb; }

    EVerboseLevel m_eVerb;

protected:

    void MyLogDetailed(const char* format, ...);
    void MyLogParanoiac(const char* format, ...);
};

#endif //_TRACER_H_

//=============================================================================
// END OF FILE
//=============================================================================
