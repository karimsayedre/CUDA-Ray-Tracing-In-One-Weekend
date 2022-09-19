#pragma once
#include <Windows.h>

#include "Log.h"

class Viewer
{
    inline static STARTUPINFO si;
    inline static PROCESS_INFORMATION pi;
public:
	static void OpenImage()
	{
        
        ZeroMemory(&si, sizeof si);
        si.cb = sizeof si;
        si.wShowWindow = TRUE;
        ZeroMemory(&pi, sizeof pi);

        wchar_t image[] = LR"("C:\Program Files\DJV2\bin\djv.exe" C:\Users\karim\Desktop\RayTracingInOneWeekend\RayTracingInOneWeekend\RayTracingInOneWeekend\Image.ppm")";
        //if (!CreateProcess(NULL, image, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi))
        {
            LOG_CORE_ERROR("Can't create process: {}", GetLastError());
        }
	}

    static void CloseImage()
    {
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
};