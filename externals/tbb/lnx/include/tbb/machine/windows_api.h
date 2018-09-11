/*
    Copyright 2005-2017 Intel Corporation.

    The source code, information and material ("Material") contained herein is owned by
    Intel Corporation or its suppliers or licensors, and title to such Material remains
    with Intel Corporation or its suppliers or licensors. The Material contains
    proprietary information of Intel or its suppliers and licensors. The Material is
    protected by worldwide copyright laws and treaty provisions. No part of the Material
    may be used, copied, reproduced, modified, published, uploaded, posted, transmitted,
    distributed or disclosed in any way without Intel's prior express written permission.
    No license under any patent, copyright or other intellectual property rights in the
    Material is granted to or conferred upon you, either expressly, by implication,
    inducement, estoppel or otherwise. Any license under such intellectual property
    rights must be express and approved by Intel in writing.

    Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
    or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
    in any way.
*/

#ifndef __TBB_machine_windows_api_H
#define __TBB_machine_windows_api_H

#if _WIN32 || _WIN64

#include <windows.h>

#if _WIN32_WINNT < 0x0600
// The following Windows API function is declared explicitly;
// otherwise it fails to compile by VS2005.
#if !defined(WINBASEAPI) || (_WIN32_WINNT < 0x0501 && _MSC_VER == 1400)
#define __TBB_WINBASEAPI extern "C"
#else
#define __TBB_WINBASEAPI WINBASEAPI
#endif
__TBB_WINBASEAPI BOOL WINAPI TryEnterCriticalSection( LPCRITICAL_SECTION );
__TBB_WINBASEAPI BOOL WINAPI InitializeCriticalSectionAndSpinCount( LPCRITICAL_SECTION, DWORD );
// Overloading WINBASEAPI macro and using local functions missing in Windows XP/2003
#define InitializeCriticalSectionEx inlineInitializeCriticalSectionEx
#define CreateSemaphoreEx inlineCreateSemaphoreEx
#define CreateEventEx inlineCreateEventEx
inline BOOL WINAPI inlineInitializeCriticalSectionEx( LPCRITICAL_SECTION lpCriticalSection, DWORD dwSpinCount, DWORD )
{
    return InitializeCriticalSectionAndSpinCount( lpCriticalSection, dwSpinCount );
}
inline HANDLE WINAPI inlineCreateSemaphoreEx( LPSECURITY_ATTRIBUTES lpSemaphoreAttributes, LONG lInitialCount, LONG lMaximumCount, LPCTSTR lpName, DWORD, DWORD )
{
    return CreateSemaphore( lpSemaphoreAttributes, lInitialCount, lMaximumCount, lpName );
}
inline HANDLE WINAPI inlineCreateEventEx( LPSECURITY_ATTRIBUTES lpEventAttributes, LPCTSTR lpName, DWORD dwFlags, DWORD )
{
    BOOL manual_reset = dwFlags&0x00000001 ? TRUE : FALSE; // CREATE_EVENT_MANUAL_RESET
    BOOL initial_set  = dwFlags&0x00000002 ? TRUE : FALSE; // CREATE_EVENT_INITIAL_SET
    return CreateEvent( lpEventAttributes, manual_reset, initial_set, lpName );
}
#endif

#if defined(RTL_SRWLOCK_INIT)
#ifndef __TBB_USE_SRWLOCK
// TODO: turn it on when bug 1952 will be fixed
#define __TBB_USE_SRWLOCK 0
#endif
#endif

#else
#error tbb/machine/windows_api.h should only be used for Windows based platforms
#endif // _WIN32 || _WIN64

#endif // __TBB_machine_windows_api_H
