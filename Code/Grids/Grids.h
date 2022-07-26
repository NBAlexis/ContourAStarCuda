#pragma once

#include <chrono>
#include <string>


inline void StartTimer(unsigned long long& uiStart)
{
    uiStart = (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

inline float StopTimer(unsigned long long uiStart)
{
    return ((std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) - uiStart) * 0.001f;
}
