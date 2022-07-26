#include "Logger.h"


Logger GLogger;

/**
*
*/
void LogOut(EVerboseLevel level, const char* format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GLogger.Print(level, format, arg);
        va_end(arg);
    }
}

/**
*
*
*/
void _LogCrucial(const char* format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GLogger.Print(EVerboseLevel::CRUCIAL, format, arg);
        GLogger.Flush();
        va_end(arg);
    }
}

/**
*
*
*/
void LogGeneral(const char* format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GLogger.Print(EVerboseLevel::GENERAL, format, arg);
        va_end(arg);
    }
}

/**
*
*
*/
void LogDetailed(const char* format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GLogger.Print(EVerboseLevel::DETAILED, format, arg);
        va_end(arg);
    }
}

/**
*
*
*/
void LogParanoiac(const char* format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GLogger.Print(EVerboseLevel::PARANOIAC, format, arg);
        va_end(arg);
    }
}

void CLogger::MyLogDetailed(const char* format, ...)
{
    if (m_eVerb < EVerboseLevel::DETAILED)
    {
        return;
    }

    va_list arg;
    {
        va_start(arg, format);
        GLogger.Print(EVerboseLevel::DETAILED, format, arg);
        va_end(arg);
    }
}

void CLogger::MyLogParanoiac(const char* format, ...)
{
    if (m_eVerb < EVerboseLevel::PARANOIAC)
    {
        return;
    }

    va_list arg;
    {
        va_start(arg, format);
        GLogger.Print(EVerboseLevel::PARANOIAC, format, arg);
        va_end(arg);
    }
}
