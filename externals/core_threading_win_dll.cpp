/* file: core_threading_win_dll.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of "stubs" for threading layer functions for win dll case.
//--
*/

#include <windows.h>
#include <stdio.h>
#include "threading.h"
#include "service_thread_pinner.h"
#include "env_detect.h"
#include "mkl_daal.h"
#include "vmlvsl.h"

static HMODULE daal_thr_dll_handle = NULL;
daal::services::Environment::LibraryThreadingType __daal_serv_get_thr_set();

#if !defined(DAAL_CHECK_DLL_SIG)
    #define DAAL_LOAD_DLL(name) LoadLibrary(name)
#else
    #define DAAL_LOAD_DLL(name) _daal_LoadLibrary(name)

    #include <Softpub.h>
    #include <wincrypt.h>
    #include <wintrust.h>

static HMODULE WINAPI _daal_LoadLibrary(LPTSTR filename)
{
    TCHAR PathBuf[MAX_PATH];
    DWORD rv;
    BOOL rv1;
    HMODULE rv2 = NULL;

    // References:
    // "Dynamic-Link Library Security" - https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-security
    // "Dynamic-Link Library Search Order" - https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order?redirectedfrom=MSDN
    // "SearchPath function" - https://docs.microsoft.com/en-us/windows/win32/api/processenv/nf-processenv-searchpathw
    // "SetSearchPathMode function" - https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setsearchpathmode
    // "SetDllDirectory function" - https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setdlldirectorya
    // "LoadLibraryExA function" - https://docs.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa

    // Exclude current directory from the serch path
    rv1 = SetDllDirectoryA("");
    if (0 == rv1)
    {
        printf("Intel DAAL FATAL ERROR: Cannot exclude current directory from serch path.\n");
        return NULL;
    }

    rv2 = LoadLibraryExA(filename, NULL, DONT_RESOLVE_DLL_REFERENCES);
    if (NULL == rv2)
    {
        printf("Intel DAAL FATAL ERROR: Cannot find/load library %s.\n", filename);
        return NULL;
    }

    rv = GetModuleFileNameA(rv2, PathBuf, MAX_PATH);
    if (0 == rv)
    {
        printf("Intel DAAL FATAL ERROR: Cannot find module %s in memory.\n", filename);
        return NULL;
    }

    FreeLibrary(rv2);
    rv2 = NULL;

    size_t strLength      = strnlen(PathBuf, MAX_PATH) + 1;
    wchar_t * wPathBuf    = new wchar_t[strLength];
    size_t convertedChars = 0;
    mbstowcs_s(&convertedChars, wPathBuf, strLength, PathBuf, _TRUNCATE);

    LONG sverif;
    DWORD lerr;
    WINTRUST_FILE_INFO fdata;
    GUID pgActionID;
    WINTRUST_DATA pWVTData;

    fdata.cbStruct       = sizeof(WINTRUST_FILE_INFO);
    fdata.pcwszFilePath  = wPathBuf;
    fdata.hFile          = NULL;
    fdata.pgKnownSubject = NULL;

    pgActionID = WINTRUST_ACTION_GENERIC_VERIFY_V2;

    pWVTData.cbStruct            = sizeof(WINTRUST_DATA);
    pWVTData.pPolicyCallbackData = NULL;
    pWVTData.pSIPClientData      = NULL;
    pWVTData.dwUIChoice          = WTD_UI_NONE;
    pWVTData.fdwRevocationChecks = WTD_REVOKE_NONE;
    pWVTData.dwUnionChoice       = WTD_CHOICE_FILE;
    pWVTData.pFile               = &fdata;
    pWVTData.dwStateAction       = WTD_STATEACTION_VERIFY;
    pWVTData.hWVTStateData       = NULL;
    pWVTData.pwszURLReference    = NULL;
    pWVTData.dwProvFlags         = 0;
    pWVTData.dwUIContext         = WTD_UICONTEXT_EXECUTE;
    pWVTData.pSignatureSettings  = NULL;

    sverif = WinVerifyTrust((HWND)INVALID_HANDLE_VALUE, &pgActionID, &pWVTData);

    switch (sverif)
    {
    case TRUST_E_NOSIGNATURE:
        lerr = GetLastError();
        if (TRUST_E_NOSIGNATURE == lerr || TRUST_E_SUBJECT_FORM_UNKNOWN == lerr || TRUST_E_PROVIDER_UNKNOWN == lerr)
        {
            printf("Intel DAAL FATAL ERROR: %s is not signed.\n", filename);
        }
        else
        {
            printf("Intel DAAL FATAL ERROR: An unknown error occurred trying toverify the signature of the %s.\n", filename);
        }
        break;

    case TRUST_E_EXPLICIT_DISTRUST: printf("Intel DAAL FATAL ERROR: The signature/publisher of %s is disallowed.\n", filename); break;

    case ERROR_SUCCESS: break;

    case TRUST_E_SUBJECT_NOT_TRUSTED: printf("Intel DAAL FATAL ERROR: The signature of %s in not trusted.\n", filename); break;

    case CRYPT_E_SECURITY_SETTINGS:
        printf("Intel DAAL FATAL ERROR: %s. The subject hash or publisher was not explicitly trusted and user trust was not allowed "
               "(CRYPT_E_SECURITY_SETTINGS).\n",
               filename);
        break;

    default: printf("Intel DAAL FATAL ERROR: %s. Error code is 0x%x.\n", filename, (unsigned int)sverif); break;
    }

    pWVTData.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(NULL, &pgActionID, &pWVTData);
    delete[] wPathBuf;

    if (ERROR_SUCCESS != sverif)
    {
        return NULL;
    }

    rv2 = LoadLibraryA(PathBuf);

    // Restore current directory from the serch path
    SetDllDirectory(NULL);

    return rv2;
}
#endif

static void load_daal_thr_dll(void)
{
    if (daal_thr_dll_handle != NULL)
    {
        return;
    }

    switch (__daal_serv_get_thr_set())
    {
    case daal::services::Environment::MultiThreaded:
    {
        daal_thr_dll_handle = DAAL_LOAD_DLL("daal_thread.dll");
        if (daal_thr_dll_handle != NULL)
        {
            return;
        }

        printf("Intel DAAL FATAL ERROR: Cannot load libdaal_thread.dll.\n");
        exit(1);

        break;
    }
    case daal::services::Environment::SingleThreaded:
    {
        daal_thr_dll_handle = DAAL_LOAD_DLL("daal_sequential.dll");
        if (daal_thr_dll_handle != NULL)
        {
            return;
        }

        printf("Intel DAAL FATAL ERROR: Cannot load libdaal_sequential.dll.\n");
        exit(1);

        break;
    }
    default:
    {
        daal_thr_dll_handle = DAAL_LOAD_DLL("daal_thread.dll");
        if (daal_thr_dll_handle != NULL)
        {
            return;
        }

        daal_thr_dll_handle = DAAL_LOAD_DLL("daal_sequential.dll");
        if (daal_thr_dll_handle != NULL)
        {
            return;
        }

        printf("Intel DAAL FATAL ERROR: Cannot load neither libdaal_thread.dll nor libdaal_sequential.dll.\n");
        exit(1);
    }
    }
}

FARPROC load_daal_thr_func(char * ordinal)
{
    FARPROC FuncAddress;

    if (daal_thr_dll_handle == NULL)
    {
        printf("Intel DAAL FATAL ERROR: Cannot load \"%s\" function because threaded layer DLL isn`t loaded.\n", ordinal);
        exit(1);
    }

    FuncAddress = GetProcAddress(daal_thr_dll_handle, ordinal);
    if (FuncAddress == NULL)
    {
        printf("Intel DAAL FATAL ERROR: Cannot load \"%s\" function.\n", ordinal);
        exit(1);
    }

    return FuncAddress;
}

typedef void * (*_threaded_malloc_t)(const size_t, const size_t);
typedef void (*_threaded_free_t)(void *);

typedef void (*_daal_threader_for_t)(int, int, const void *, daal::functype);
typedef void (*_daal_threader_for_blocked_t)(int, int, const void *, daal::functype2);
typedef int (*_daal_threader_get_max_threads_t)(void);

typedef void * (*_daal_get_tls_ptr_t)(void *, daal::tls_functype);
typedef void (*_daal_del_tls_ptr_t)(void *);
typedef void * (*_daal_get_tls_local_t)(void *);
typedef void (*_daal_reduce_tls_t)(void *, void *, daal::tls_reduce_functype);
typedef void (*_daal_parallel_reduce_tls_t)(void *, void *, daal::tls_reduce_functype);

typedef void * (*_daal_get_ls_ptr_t)(void *, daal::tls_functype);
typedef void (*_daal_del_ls_ptr_t)(void *);
typedef void * (*_daal_get_ls_local_t)(void *);
typedef void (*_daal_release_ls_local_t)(void *, void *);
typedef void (*_daal_reduce_ls_t)(void *, void *, daal::tls_reduce_functype);

typedef void * (*_daal_new_mutex_t)();
typedef void (*_daal_del_mutex_t)(void *);
typedef void (*_daal_lock_mutex_t)(void *);
typedef void (*_daal_unlock_mutex_t)(void *);

typedef void * (*_daal_new_task_group_t)();
typedef void (*_daal_del_task_group_t)(void * taskGroupPtr);
typedef void (*_daal_run_task_group_t)(void * taskGroupPtr, daal::task * t);
typedef void (*_daal_wait_task_group_t)(void * taskGroupPtr);

typedef bool (*_daal_is_in_parallel_t)();
typedef void (*_daal_tbb_task_scheduler_free_t)(void *& init);
typedef size_t (*_setNumberOfThreads_t)(const size_t, void **);
typedef void * (*_daal_threader_env_t)();

#if !(defined DAAL_THREAD_PINNING_DISABLED)
typedef void (*_thread_pinner_thread_pinner_init_t)();
typedef void (*_thread_pinner_read_topology_t)();
typedef void (*_thread_pinner_on_scheduler_entry_t)(bool);
typedef void (*_thread_pinner_on_scheduler_exit_t)(bool);
typedef void (*_thread_pinner_execute_t)(daal::services::internal::thread_pinner_task_t & f);
typedef int (*_thread_pinner_get_status_t)();
typedef bool (*_thread_pinner_get_pinning_t)();
typedef bool (*_thread_pinner_set_pinning_t)(bool p);
typedef void * (*_getThreadPinner_t)(bool create_pinner, void (*read_topo)(int &, int &, int &, int **), void (*deleter)(void *));
#endif

static _threaded_malloc_t _threaded_malloc_ptr = NULL;
static _threaded_free_t _threaded_free_ptr     = NULL;

static _daal_threader_for_t _daal_threader_for_ptr                         = NULL;
static _daal_threader_for_blocked_t _daal_threader_for_blocked_ptr         = NULL;
static _daal_threader_for_t _daal_threader_for_optional_ptr                = NULL;
static _daal_threader_get_max_threads_t _daal_threader_get_max_threads_ptr = NULL;

static _daal_get_tls_ptr_t _daal_get_tls_ptr_ptr                 = NULL;
static _daal_del_tls_ptr_t _daal_del_tls_ptr_ptr                 = NULL;
static _daal_get_tls_local_t _daal_get_tls_local_ptr             = NULL;
static _daal_reduce_tls_t _daal_reduce_tls_ptr                   = NULL;
static _daal_parallel_reduce_tls_t _daal_parallel_reduce_tls_ptr = NULL;

static _daal_get_ls_ptr_t _daal_get_ls_ptr_ptr             = NULL;
static _daal_del_ls_ptr_t _daal_del_ls_ptr_ptr             = NULL;
static _daal_get_ls_local_t _daal_get_ls_local_ptr         = NULL;
static _daal_release_ls_local_t _daal_release_ls_local_ptr = NULL;
static _daal_reduce_tls_t _daal_reduce_ls_ptr              = NULL;

static _daal_new_mutex_t _daal_new_mutex_ptr     = NULL;
static _daal_del_mutex_t _daal_del_mutex_ptr     = NULL;
static _daal_lock_mutex_t _daal_lock_mutex_ptr   = NULL;
static _daal_lock_mutex_t _daal_unlock_mutex_ptr = NULL;

static _daal_new_task_group_t _daal_new_task_group_ptr   = NULL;
static _daal_del_task_group_t _daal_del_task_group_ptr   = NULL;
static _daal_run_task_group_t _daal_run_task_group_ptr   = NULL;
static _daal_wait_task_group_t _daal_wait_task_group_ptr = NULL;

static _daal_is_in_parallel_t _daal_is_in_parallel_ptr                   = NULL;
static _daal_tbb_task_scheduler_free_t _daal_tbb_task_scheduler_free_ptr = NULL;
static _setNumberOfThreads_t _setNumberOfThreads_ptr                     = NULL;
static _daal_threader_env_t _daal_threader_env_ptr                       = NULL;

#if !(defined DAAL_THREAD_PINNING_DISABLED)
static _thread_pinner_thread_pinner_init_t _thread_pinner_thread_pinner_init_ptr = NULL;
static _thread_pinner_read_topology_t _thread_pinner_read_topology_ptr           = NULL;
static _thread_pinner_on_scheduler_entry_t _thread_pinner_on_scheduler_entry_ptr = NULL;
static _thread_pinner_on_scheduler_exit_t _thread_pinner_on_scheduler_exit_ptr   = NULL;
static _thread_pinner_execute_t _thread_pinner_execute_ptr                       = NULL;
static _thread_pinner_get_status_t _thread_pinner_get_status_ptr                 = NULL;
static _thread_pinner_get_pinning_t _thread_pinner_get_pinning_ptr               = NULL;
static _thread_pinner_set_pinning_t _thread_pinner_set_pinning_ptr               = NULL;
static _getThreadPinner_t _getThreadPinner_ptr                                   = NULL;
#endif

DAAL_EXPORT void * _threaded_scalable_malloc(const size_t size, const size_t alignment)
{
    load_daal_thr_dll();
    if (_threaded_malloc_ptr == NULL)
    {
        _threaded_malloc_ptr = (_threaded_malloc_t)load_daal_thr_func("_threaded_scalable_malloc");
    }
    return _threaded_malloc_ptr(size, alignment);
}

DAAL_EXPORT void _threaded_scalable_free(void * ptr)
{
    load_daal_thr_dll();
    if (_threaded_free_ptr == NULL)
    {
        _threaded_free_ptr = (_threaded_free_t)load_daal_thr_func("_threaded_scalable_free");
    }
    _threaded_free_ptr(ptr);
}

DAAL_EXPORT void _daal_threader_for(int n, int threads_request, const void * a, daal::functype func)
{
    load_daal_thr_dll();
    if (_daal_threader_for_ptr == NULL)
    {
        _daal_threader_for_ptr = (_daal_threader_for_t)load_daal_thr_func("_daal_threader_for");
    }
    _daal_threader_for_ptr(n, threads_request, a, func);
}

DAAL_EXPORT void _daal_threader_for_blocked(int n, int threads_request, const void * a, daal::functype2 func)
{
    load_daal_thr_dll();
    if (_daal_threader_for_blocked_ptr == NULL)
    {
        _daal_threader_for_blocked_ptr = (_daal_threader_for_blocked_t)load_daal_thr_func("_daal_threader_for_blocked");
    }
    _daal_threader_for_blocked_ptr(n, threads_request, a, func);
}

DAAL_EXPORT void _daal_threader_for_optional(int n, int threads_request, const void * a, daal::functype func)
{
    load_daal_thr_dll();
    if (_daal_threader_for_optional_ptr == NULL)
    {
        _daal_threader_for_optional_ptr = (_daal_threader_for_t)load_daal_thr_func("_daal_threader_for_optional");
    }
    _daal_threader_for_optional_ptr(n, threads_request, a, func);
}

DAAL_EXPORT int _daal_threader_get_max_threads()
{
    load_daal_thr_dll();
    if (_daal_threader_get_max_threads_ptr == NULL)
    {
        _daal_threader_get_max_threads_ptr = (_daal_threader_get_max_threads_t)load_daal_thr_func("_daal_threader_get_max_threads");
    }
    return _daal_threader_get_max_threads_ptr();
}

DAAL_EXPORT void * _daal_get_tls_ptr(void * a, daal::tls_functype func)
{
    load_daal_thr_dll();
    if (_daal_get_tls_ptr_ptr == NULL)
    {
        _daal_get_tls_ptr_ptr = (_daal_get_tls_ptr_t)load_daal_thr_func("_daal_get_tls_ptr");
    }
    return _daal_get_tls_ptr_ptr(a, func);
}

DAAL_EXPORT void _daal_del_tls_ptr(void * tlsPtr)
{
    load_daal_thr_dll();
    if (_daal_del_tls_ptr_ptr == NULL)
    {
        _daal_del_tls_ptr_ptr = (_daal_del_tls_ptr_t)load_daal_thr_func("_daal_del_tls_ptr");
    }
    _daal_del_tls_ptr_ptr(tlsPtr);
}

DAAL_EXPORT void * _daal_get_tls_local(void * tlsPtr)
{
    load_daal_thr_dll();
    if (_daal_get_tls_local_ptr == NULL)
    {
        _daal_get_tls_local_ptr = (_daal_get_tls_local_t)load_daal_thr_func("_daal_get_tls_local");
    }
    return _daal_get_tls_local_ptr(tlsPtr);
}

DAAL_EXPORT void _daal_reduce_tls(void * tlsPtr, void * a, daal::tls_reduce_functype func)
{
    load_daal_thr_dll();
    if (_daal_reduce_tls_ptr == NULL)
    {
        _daal_reduce_tls_ptr = (_daal_reduce_tls_t)load_daal_thr_func("_daal_reduce_tls");
    }
    _daal_reduce_tls_ptr(tlsPtr, a, func);
}

DAAL_EXPORT void _daal_parallel_reduce_tls(void * tlsPtr, void * a, daal::tls_reduce_functype func)
{
    load_daal_thr_dll();
    if (_daal_parallel_reduce_tls_ptr == NULL)
    {
        _daal_parallel_reduce_tls_ptr = (_daal_parallel_reduce_tls_t)load_daal_thr_func("_daal_parallel_reduce_tls");
    }
    _daal_parallel_reduce_tls_ptr(tlsPtr, a, func);
}

DAAL_EXPORT void * _daal_get_ls_ptr(void * a, daal::tls_functype func)
{
    load_daal_thr_dll();
    if (_daal_get_ls_ptr_ptr == NULL)
    {
        _daal_get_ls_ptr_ptr = (_daal_get_ls_ptr_t)load_daal_thr_func("_daal_get_ls_ptr");
    }
    return _daal_get_ls_ptr_ptr(a, func);
}

DAAL_EXPORT void _daal_del_ls_ptr(void * lsPtr)
{
    load_daal_thr_dll();
    if (_daal_del_ls_ptr_ptr == NULL)
    {
        _daal_del_ls_ptr_ptr = (_daal_del_ls_ptr_t)load_daal_thr_func("_daal_del_ls_ptr");
    }
    _daal_del_ls_ptr_ptr(lsPtr);
}

DAAL_EXPORT void * _daal_get_ls_local(void * lsPtr)
{
    load_daal_thr_dll();
    if (_daal_get_ls_local_ptr == NULL)
    {
        _daal_get_ls_local_ptr = (_daal_get_ls_local_t)load_daal_thr_func("_daal_get_ls_local");
    }
    return _daal_get_ls_local_ptr(lsPtr);
}

DAAL_EXPORT void _daal_release_ls_local(void * lsPtr, void * a)
{
    load_daal_thr_dll();
    if (_daal_release_ls_local_ptr == NULL)
    {
        _daal_release_ls_local_ptr = (_daal_release_ls_local_t)load_daal_thr_func("_daal_release_ls_local");
    }
    _daal_release_ls_local_ptr(lsPtr, a);
}

DAAL_EXPORT void _daal_reduce_ls(void * lsPtr, void * a, daal::tls_reduce_functype func)
{
    load_daal_thr_dll();
    if (_daal_reduce_ls_ptr == NULL)
    {
        _daal_reduce_ls_ptr = (_daal_reduce_ls_t)load_daal_thr_func("_daal_reduce_ls");
    }
    _daal_reduce_ls_ptr(lsPtr, a, func);
}

DAAL_EXPORT void * _daal_new_mutex()
{
    load_daal_thr_dll();
    if (_daal_new_mutex_ptr == NULL)
    {
        _daal_new_mutex_ptr = (_daal_new_mutex_t)load_daal_thr_func("_daal_new_mutex");
    }
    return _daal_new_mutex_ptr();
}

DAAL_EXPORT void _daal_lock_mutex(void * mutexPtr)
{
    load_daal_thr_dll();
    if (_daal_lock_mutex_ptr == NULL)
    {
        _daal_lock_mutex_ptr = (_daal_lock_mutex_t)load_daal_thr_func("_daal_lock_mutex");
    }
    _daal_lock_mutex_ptr(mutexPtr);
}

DAAL_EXPORT void _daal_unlock_mutex(void * mutexPtr)
{
    load_daal_thr_dll();
    if (_daal_unlock_mutex_ptr == NULL)
    {
        _daal_unlock_mutex_ptr = (_daal_unlock_mutex_t)load_daal_thr_func("_daal_unlock_mutex");
    }
    _daal_unlock_mutex_ptr(mutexPtr);
}

DAAL_EXPORT void _daal_del_mutex(void * mutexPtr)
{
    load_daal_thr_dll();
    if (_daal_del_mutex_ptr == NULL)
    {
        _daal_del_mutex_ptr = (_daal_del_mutex_t)load_daal_thr_func("_daal_del_mutex");
    }
    _daal_del_mutex_ptr(mutexPtr);
}

DAAL_EXPORT void * _daal_new_task_group()
{
    load_daal_thr_dll();
    if (_daal_new_task_group_ptr == NULL)
    {
        _daal_new_task_group_ptr = (_daal_new_task_group_t)load_daal_thr_func("_daal_new_task_group");
    }
    return _daal_new_task_group_ptr();
}

DAAL_EXPORT void _daal_del_task_group(void * taskGroupPtr)
{
    load_daal_thr_dll();
    if (_daal_del_task_group_ptr == NULL)
    {
        _daal_del_task_group_ptr = (_daal_del_task_group_t)load_daal_thr_func("_daal_del_task_group");
    }
    _daal_del_task_group_ptr(taskGroupPtr);
}

DAAL_EXPORT void _daal_run_task_group(void * taskGroupPtr, daal::task * t)
{
    load_daal_thr_dll();
    if (_daal_run_task_group_ptr == NULL)
    {
        _daal_run_task_group_ptr = (_daal_run_task_group_t)load_daal_thr_func("_daal_run_task_group");
    }
    _daal_run_task_group_ptr(taskGroupPtr, t);
}

DAAL_EXPORT void _daal_wait_task_group(void * taskGroupPtr)
{
    load_daal_thr_dll();
    if (_daal_wait_task_group_ptr == NULL)
    {
        _daal_wait_task_group_ptr = (_daal_wait_task_group_t)load_daal_thr_func("_daal_wait_task_group");
    }
    _daal_wait_task_group_ptr(taskGroupPtr);
}

DAAL_EXPORT bool _daal_is_in_parallel()
{
    load_daal_thr_dll();
    if (_daal_is_in_parallel_ptr == NULL)
    {
        _daal_is_in_parallel_ptr = (_daal_is_in_parallel_t)load_daal_thr_func("_daal_is_in_parallel");
    }
    return _daal_is_in_parallel_ptr();
}

DAAL_EXPORT void _daal_tbb_task_scheduler_free(void *& init)
{
    load_daal_thr_dll();
    if (_daal_tbb_task_scheduler_free_ptr == NULL)
    {
        _daal_tbb_task_scheduler_free_ptr = (_daal_tbb_task_scheduler_free_t)load_daal_thr_func("_daal_tbb_task_scheduler_free");
    }
    return _daal_tbb_task_scheduler_free_ptr(init);
}

DAAL_EXPORT size_t _setNumberOfThreads(const size_t numThreads, void ** init)
{
    load_daal_thr_dll();
    if (_setNumberOfThreads_ptr == NULL)
    {
        _setNumberOfThreads_ptr = (_setNumberOfThreads_t)load_daal_thr_func("_setNumberOfThreads");
    }
    return _setNumberOfThreads_ptr(numThreads, init);
}

DAAL_EXPORT void * _daal_threader_env()
{
    load_daal_thr_dll();
    if (_daal_threader_env_ptr == NULL)
    {
        _daal_threader_env_ptr = (_daal_threader_env_t)load_daal_thr_func("_daal_threader_env");
    }
    return _daal_threader_env_ptr();
}

#if !(defined DAAL_THREAD_PINNING_DISABLED)
DAAL_EXPORT void _thread_pinner_thread_pinner_init()
{
    load_daal_thr_dll();
    if (_thread_pinner_thread_pinner_init_ptr == NULL)
    {
        _thread_pinner_thread_pinner_init_ptr = (_thread_pinner_thread_pinner_init_t)load_daal_thr_func("_thread_pinner_thread_pinner_init");
    }
    _thread_pinner_thread_pinner_init_ptr();
}

DAAL_EXPORT void _thread_pinner_read_topology()
{
    load_daal_thr_dll();
    if (_thread_pinner_read_topology_ptr == NULL)
    {
        _thread_pinner_read_topology_ptr = (_thread_pinner_read_topology_t)load_daal_thr_func("_thread_pinner_read_topology");
    }
    _thread_pinner_read_topology_ptr();
}

DAAL_EXPORT void _thread_pinner_on_scheduler_entry(bool p)
{
    load_daal_thr_dll();
    if (_thread_pinner_on_scheduler_entry_ptr == NULL)
    {
        _thread_pinner_on_scheduler_entry_ptr = (_thread_pinner_on_scheduler_entry_t)load_daal_thr_func("_thread_pinner_on_scheduler_entry");
    }
    _thread_pinner_on_scheduler_entry_ptr(p);
}

DAAL_EXPORT void _thread_pinner_on_scheduler_exit(bool p)
{
    load_daal_thr_dll();
    if (_thread_pinner_on_scheduler_exit_ptr == NULL)
    {
        _thread_pinner_on_scheduler_exit_ptr = (_thread_pinner_on_scheduler_exit_t)load_daal_thr_func("_thread_pinner_on_scheduler_exit");
    }
    _thread_pinner_on_scheduler_exit_ptr(p);
}

DAAL_EXPORT void _thread_pinner_execute(daal::services::internal::thread_pinner_task_t & task)
{
    load_daal_thr_dll();
    if (_thread_pinner_execute_ptr == NULL)
    {
        _thread_pinner_execute_ptr = (_thread_pinner_execute_t)load_daal_thr_func("_thread_pinner_execute");
    }
    _thread_pinner_execute_ptr(task);
}

DAAL_EXPORT int _thread_pinner_get_status()
{
    load_daal_thr_dll();
    if (_thread_pinner_get_status_ptr == NULL)
    {
        _thread_pinner_get_status_ptr = (_thread_pinner_get_status_t)load_daal_thr_func("_thread_pinner_get_status");
    }
    return _thread_pinner_get_status_ptr();
}

DAAL_EXPORT bool _thread_pinner_get_pinning()
{
    load_daal_thr_dll();
    if (_thread_pinner_get_pinning_ptr == NULL)
    {
        _thread_pinner_get_pinning_ptr = (_thread_pinner_get_pinning_t)load_daal_thr_func("_thread_pinner_get_pinning");
    }
    return _thread_pinner_get_pinning_ptr();
}

DAAL_EXPORT bool _thread_pinner_set_pinning(bool p)
{
    load_daal_thr_dll();
    if (_thread_pinner_set_pinning_ptr == NULL)
    {
        _thread_pinner_set_pinning_ptr = (_thread_pinner_set_pinning_t)load_daal_thr_func("_thread_pinner_set_pinning");
    }
    return _thread_pinner_set_pinning_ptr(p);
}

DAAL_EXPORT void * _getThreadPinner(bool create_pinner, void (*read_topo)(int &, int &, int &, int **), void (*deleter)(void *))
{
    load_daal_thr_dll();
    if (_getThreadPinner_ptr == NULL)
    {
        _getThreadPinner_ptr = (_getThreadPinner_t)load_daal_thr_func("_getThreadPinner");
    }
    return _getThreadPinner_ptr(create_pinner, read_topo, deleter);
}
#endif

#define CALL_VOID_FUNC_FROM_DLL(fn_dpref, fn_name, argdecl, argcall)                  \
    typedef void(*##fn_dpref##fn_name##_t)##argdecl;                                  \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr = NULL;                      \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, avx512_, fn_name, argdecl, argcall)         \
    CALL_VOID_FUNC_FROM_DLL_CPU_MIC(fn_dpref, avx512_mic_, fn_name, argdecl, argcall) \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, avx2_, fn_name, argdecl, argcall)           \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, avx_, fn_name, argdecl, argcall)            \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, sse42_, fn_name, argdecl, argcall)          \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, ssse3_, fn_name, argdecl, argcall)          \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, sse2_, fn_name, argdecl, argcall)

#define CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, fn_cpu, fn_name, argdecl, argcall)                                 \
    void fn_dpref##fn_cpu##fn_name##argdecl                                                                      \
    {                                                                                                            \
        load_daal_thr_dll();                                                                                     \
        if (##fn_dpref##fn_name##_ptr == NULL)                                                                   \
        {                                                                                                        \
            ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref #fn_cpu #fn_name); \
        }                                                                                                        \
        ##fn_dpref##fn_name##_ptr##argcall;                                                                      \
    }

#if defined(_WIN64)
    #define CALL_VOID_FUNC_FROM_DLL_CPU_MIC(fn_dpref, fn_cpu, fn_name, argdecl, argcall)                             \
        void fn_dpref##fn_cpu##fn_name##argdecl                                                                      \
        {                                                                                                            \
            load_daal_thr_dll();                                                                                     \
            if (##fn_dpref##fn_name##_ptr == NULL)                                                                   \
            {                                                                                                        \
                ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref #fn_cpu #fn_name); \
            }                                                                                                        \
            ##fn_dpref##fn_name##_ptr##argcall;                                                                      \
        }
#else
    #define CALL_VOID_FUNC_FROM_DLL_CPU_MIC(fn_dpref, fn_cpu, fn_name, argdecl, argcall)
#endif

#define CALL_RET_FUNC_FROM_DLL(ret_type, fn_dpref, fn_name, argdecl, argcall)                  \
    typedef ret_type(*##fn_dpref##fn_name##_t)##argdecl;                                       \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr = NULL;                               \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, avx512_, fn_name, argdecl, argcall)         \
    CALL_RET_FUNC_FROM_DLL_CPU_MIC(ret_type, fn_dpref, avx512_mic_, fn_name, argdecl, argcall) \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, avx2_, fn_name, argdecl, argcall)           \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, avx_, fn_name, argdecl, argcall)            \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, sse42_, fn_name, argdecl, argcall)          \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, ssse3_, fn_name, argdecl, argcall)          \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, sse2_, fn_name, argdecl, argcall)

#define CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, fn_cpu, fn_name, argdecl, argcall)                        \
    ret_type fn_dpref##fn_cpu##fn_name##argdecl                                                                  \
    {                                                                                                            \
        load_daal_thr_dll();                                                                                     \
        if (##fn_dpref##fn_name##_ptr == NULL)                                                                   \
        {                                                                                                        \
            ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref #fn_cpu #fn_name); \
        }                                                                                                        \
        return fn_dpref##fn_name##_ptr##argcall;                                                                 \
    }

#if defined(_WIN64)
    #define CALL_RET_FUNC_FROM_DLL_CPU_MIC(ret_type, fn_dpref, fn_cpu, fn_name, argdecl, argcall)                    \
        ret_type fn_dpref##fn_cpu##fn_name##argdecl                                                                  \
        {                                                                                                            \
            load_daal_thr_dll();                                                                                     \
            if (##fn_dpref##fn_name##_ptr == NULL)                                                                   \
            {                                                                                                        \
                ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref #fn_cpu #fn_name); \
            }                                                                                                        \
            return fn_dpref##fn_name##_ptr##argcall;                                                                 \
        }
#else
    #define CALL_RET_FUNC_FROM_DLL_CPU_MIC(ret_type, fn_dpref, fn_cpu, fn_name, argdecl, argcall)
#endif

/* Used directly in Intel DAAL */
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dsyrk,
                        (const char * uplo, const char * trans, const DAAL_INT * n, const DAAL_INT * k, const double * alpha, const double * a,
                         const DAAL_INT * lda, const double * beta, double * c, const DAAL_INT * ldc),
                        (uplo, trans, n, k, alpha, a, lda, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, ssyrk,
                        (const char * uplo, const char * trans, const DAAL_INT * n, const DAAL_INT * k, const float * alpha, const float * a,
                         const DAAL_INT * lda, const float * beta, float * c, const DAAL_INT * ldc),
                        (uplo, trans, n, k, alpha, a, lda, beta, c, ldc));

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dsyr,
                        (const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                         const DAAL_INT * lda),
                        (uplo, n, alpha, x, incx, a, lda));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, ssyr,
                        (const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                         const DAAL_INT * lda),
                        (uplo, n, alpha, x, incx, a, lda));

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dgemm,
                        (const char * transa, const char * transb, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const double * alpha,
                         const double * a, const DAAL_INT * lda, const double * b, const DAAL_INT * ldb, const double * beta, double * c,
                         const DAAL_INT * ldc),
                        (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, sgemm,
                        (const char * transa, const char * transb, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const float * alpha,
                         const float * a, const DAAL_INT * lda, const float * b, const DAAL_INT * ldb, const float * beta, float * c,
                         const DAAL_INT * ldc),
                        (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xdgemm,
                        (const char * transa, const char * transb, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const double * alpha,
                         const double * a, const DAAL_INT * lda, const double * b, const DAAL_INT * ldb, const double * beta, double * c,
                         const DAAL_INT * ldc),
                        (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xsgemm,
                        (const char * transa, const char * transb, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const float * alpha,
                         const float * a, const DAAL_INT * lda, const float * b, const DAAL_INT * ldb, const float * beta, float * c,
                         const DAAL_INT * ldc),
                        (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dsymm,
                        (const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a,
                         const DAAL_INT * lda, const double * b, const DAAL_INT * ldb, const double * beta, double * c, const DAAL_INT * ldc),
                        (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, ssymm,
                        (const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a,
                         const DAAL_INT * lda, const float * b, const DAAL_INT * ldb, const float * beta, float * c, const DAAL_INT * ldc),
                        (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc));

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dgemv,
                        (const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                         const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy),
                        (trans, m, n, alpha, a, lda, x, incx, beta, y, incy));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, sgemv,
                        (const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                         const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy),
                        (trans, m, n, alpha, a, lda, x, incx, beta, y, incy));

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, daxpy,
                        (const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * y, const DAAL_INT * incy),
                        (n, alpha, x, incx, y, incy));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, saxpy,
                        (const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * y, const DAAL_INT * incy),
                        (n, alpha, x, incx, y, incy));

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xdsyr,
                        (const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                         const DAAL_INT * lda),
                        (uplo, n, alpha, x, incx, a, lda));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xssyr,
                        (const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                         const DAAL_INT * lda),
                        (uplo, n, alpha, x, incx, a, lda));

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xdsyrk,
                        (const char * uplo, const char * trans, const DAAL_INT * n, const DAAL_INT * k, const double * alpha, const double * a,
                         const DAAL_INT * lda, const double * beta, double * c, const DAAL_INT * ldc),
                        (uplo, trans, n, k, alpha, a, lda, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xssyrk,
                        (const char * uplo, const char * trans, const DAAL_INT * n, const DAAL_INT * k, const float * alpha, const float * a,
                         const DAAL_INT * lda, const float * beta, float * c, const DAAL_INT * ldc),
                        (uplo, trans, n, k, alpha, a, lda, beta, c, ldc));

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xdsymm,
                        (const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a,
                         const DAAL_INT * lda, const double * b, const DAAL_INT * ldb, const double * beta, double * c, const DAAL_INT * ldc),
                        (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xssymm,
                        (const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a,
                         const DAAL_INT * lda, const float * b, const DAAL_INT * ldb, const float * beta, float * c, const DAAL_INT * ldc),
                        (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc));

CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_dcsrmultd,
                        (const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, double * a, DAAL_INT * ja, DAAL_INT * ia,
                         double * b, DAAL_INT * jb, DAAL_INT * ib, double * c, DAAL_INT * ldc),
                        (transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_scsrmultd,
                        (const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, float * a, DAAL_INT * ja, DAAL_INT * ia,
                         float * b, DAAL_INT * jb, DAAL_INT * ib, float * c, DAAL_INT * ldc),
                        (transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc));

CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_dcsrmv,
                        (const char * transa, const DAAL_INT * m, const DAAL_INT * k, const double * alpha, const char * matdescra,
                         const double * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const double * x,
                         const double * beta, double * y),
                        (transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y));
CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_scsrmv,
                        (const char * transa, const DAAL_INT * m, const DAAL_INT * k, const float * alpha, const char * matdescra, const float * val,
                         const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const float * x, const float * beta, float * y),
                        (transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dpotrf, (const char * uplo, const DAAL_INT * n, double * a, const DAAL_INT * lda, DAAL_INT * info, int iuplo),
                        (uplo, n, a, lda, info, iuplo));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, spotrf, (const char * uplo, const DAAL_INT * n, float * a, const DAAL_INT * lda, DAAL_INT * info, int iuplo),
                        (uplo, n, a, lda, info, iuplo));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dpotrs,
                        (const char * uplo, const DAAL_INT * n, const DAAL_INT * nrhs, const double * a, const DAAL_INT * lda, double * b,
                         const DAAL_INT * ldb, DAAL_INT * info, int iuplo),
                        (uplo, n, nrhs, a, lda, b, ldb, info, iuplo));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, spotrs,
                        (const char * uplo, const DAAL_INT * n, const DAAL_INT * nrhs, const float * a, const DAAL_INT * lda, float * b,
                         const DAAL_INT * ldb, DAAL_INT * info, int iuplo),
                        (uplo, n, nrhs, a, lda, b, ldb, info, iuplo));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dpotri, (const char * uplo, const DAAL_INT * n, double * a, const DAAL_INT * lda, DAAL_INT * info, int iuplo),
                        (uplo, n, a, lda, info, iuplo));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, spotri, (const char * uplo, const DAAL_INT * n, float * a, const DAAL_INT * lda, DAAL_INT * info, int iuplo),
                        (uplo, n, a, lda, info, iuplo));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgerqf,
                        (const DAAL_INT * m, const DAAL_INT * n, double * a, const DAAL_INT * lda, double * tau, double * work,
                         const DAAL_INT * lwork, DAAL_INT * info),
                        (m, n, a, lda, tau, work, lwork, info));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgerqf,
                        (const DAAL_INT * m, const DAAL_INT * n, float * a, const DAAL_INT * lda, float * tau, float * work, const DAAL_INT * lwork,
                         DAAL_INT * info),
                        (m, n, a, lda, tau, work, lwork, info));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dormrq,
                        (const char * side, const char * trans, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const double * a,
                         const DAAL_INT * lda, const double * tau, double * c, const DAAL_INT * ldc, double * work, const DAAL_INT * lwork,
                         DAAL_INT * info, int iside, int itrans),
                        (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, iside, itrans));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sormrq,
                        (const char * side, const char * trans, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const float * a,
                         const DAAL_INT * lda, const float * tau, float * c, const DAAL_INT * ldc, float * work, const DAAL_INT * lwork,
                         DAAL_INT * info, int iside, int itrans),
                        (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, iside, itrans));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dormqr,
                        (const char * side, const char * trans, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const double * a,
                         const DAAL_INT * lda, const double * tau, double * c, const DAAL_INT * ldc, double * work, const DAAL_INT * lwork,
                         DAAL_INT * info, int iside, int itrans),
                        (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, iside, itrans));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sormqr,
                        (const char * side, const char * trans, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const float * a,
                         const DAAL_INT * lda, const float * tau, float * c, const DAAL_INT * ldc, float * work, const DAAL_INT * lwork,
                         DAAL_INT * info, int iside, int itrans),
                        (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, iside, itrans));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dtrtrs,
                        (const char * uplo, const char * trans, const char * diag, const DAAL_INT * n, const DAAL_INT * nrhs, const double * a,
                         const DAAL_INT * lda, double * b, const DAAL_INT * ldb, DAAL_INT * info, int iuplo, int itrans, int idiag),
                        (uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, iuplo, itrans, idiag));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, strtrs,
                        (const char * uplo, const char * trans, const char * diag, const DAAL_INT * n, const DAAL_INT * nrhs, const float * a,
                         const DAAL_INT * lda, float * b, const DAAL_INT * ldb, DAAL_INT * info, int iuplo, int itrans, int idiag),
                        (uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, iuplo, itrans, idiag));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dpptrf, (const char * uplo, const DAAL_INT * n, double * ap, DAAL_INT * info, int iuplo),
                        (uplo, n, ap, info, iuplo));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, spptrf, (const char * uplo, const DAAL_INT * n, float * ap, DAAL_INT * info, int iuplo),
                        (uplo, n, ap, info, iuplo));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgeqrf,
                        (const DAAL_INT * m, const DAAL_INT * n, double * a, const DAAL_INT * lda, double * tau, double * work,
                         const DAAL_INT * lwork, DAAL_INT * info),
                        (m, n, a, lda, tau, work, lwork, info));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgeqrf,
                        (const DAAL_INT * m, const DAAL_INT * n, float * a, const DAAL_INT * lda, float * tau, float * work, const DAAL_INT * lwork,
                         DAAL_INT * info),
                        (m, n, a, lda, tau, work, lwork, info));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgeqp3,
                        (const DAAL_INT * m, const DAAL_INT * n, double * a, const DAAL_INT * lda, DAAL_INT * jpvt, double * tau, double * work,
                         const DAAL_INT * lwork, DAAL_INT * info),
                        (m, n, a, lda, jpvt, tau, work, lwork, info));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgeqp3,
                        (const DAAL_INT * m, const DAAL_INT * n, float * a, const DAAL_INT * lda, DAAL_INT * jpvt, float * tau, float * work,
                         const DAAL_INT * lwork, DAAL_INT * info),
                        (m, n, a, lda, jpvt, tau, work, lwork, info));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dorgqr,
                        (const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, double * a, const DAAL_INT * lda, const double * tau,
                         double * work, const DAAL_INT * lwork, DAAL_INT * info),
                        (m, n, k, a, lda, tau, work, lwork, info));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sorgqr,
                        (const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, float * a, const DAAL_INT * lda, const float * tau, float * work,
                         const DAAL_INT * lwork, DAAL_INT * info),
                        (m, n, k, a, lda, tau, work, lwork, info));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgesvd,
                        (const char * jobu, const char * jobvt, const DAAL_INT * m, const DAAL_INT * n, double * a, const DAAL_INT * lda, double * s,
                         double * u, const DAAL_INT * ldu, double * vt, const DAAL_INT * ldvt, double * work, const DAAL_INT * lwork, DAAL_INT * info,
                         int ijobu, int ijobvt),
                        (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info, ijobu, ijobvt));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgesvd,
                        (const char * jobu, const char * jobvt, const DAAL_INT * m, const DAAL_INT * n, float * a, const DAAL_INT * lda, float * s,
                         float * u, const DAAL_INT * ldu, float * vt, const DAAL_INT * ldvt, float * work, const DAAL_INT * lwork, DAAL_INT * info,
                         int ijobu, int ijobvt),
                        (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info, ijobu, ijobvt));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dsyevd,
                        (const char * jobz, const char * uplo, const DAAL_INT * n, double * a, const DAAL_INT * lda, double * w, double * work,
                         const DAAL_INT * lwork, DAAL_INT * iwork, const DAAL_INT * liwork, DAAL_INT * info, int ijobz, int iuplo),
                        (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, ijobz, iuplo));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, ssyevd,
                        (const char * jobz, const char * uplo, const DAAL_INT * n, float * a, const DAAL_INT * lda, float * w, float * work,
                         const DAAL_INT * lwork, DAAL_INT * iwork, const DAAL_INT * liwork, DAAL_INT * info, int ijobz, int iuplo),
                        (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, ijobz, iuplo));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dsyev,
                        (const char * jobz, const char * uplo, const DAAL_INT * n, double * a, const DAAL_INT * lda, double * w, double * work,
                         const DAAL_INT * lwork, DAAL_INT * info, int ijobz, int iuplo),
                        (jobz, uplo, n, a, lda, w, work, lwork, info, ijobz, iuplo));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, ssyev,
                        (const char * jobz, const char * uplo, const DAAL_INT * n, float * a, const DAAL_INT * lda, float * w, float * work,
                         const DAAL_INT * lwork, DAAL_INT * info, int ijobz, int iuplo),
                        (jobz, uplo, n, a, lda, w, work, lwork, info, ijobz, iuplo));

CALL_RET_FUNC_FROM_DLL(double, fpk_blas_, xddot,
                       (const DAAL_INT * n, const double * x, const DAAL_INT * incx, const double * y, const DAAL_INT * incy), (n, x, incx, y, incy));
CALL_RET_FUNC_FROM_DLL(float, fpk_blas_, xsdot, (const DAAL_INT * n, const float * x, const DAAL_INT * incx, const float * y, const DAAL_INT * incy),
                       (n, x, incx, y, incy));

#define CSRMM_ARGS(FPTYPE)                                                                                                                       \
    const char *transa, const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const FPTYPE *alpha, const char *matdescra, const FPTYPE *val,  \
        const DAAL_INT *indx, const DAAL_INT *pntrb, const DAAL_INT *pntre, const FPTYPE *b, const DAAL_INT *ldb, const FPTYPE *beta, FPTYPE *c, \
        const DAAL_INT *ldc

CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_scsrmm, (CSRMM_ARGS(float)),
                        (transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_dcsrmm, (CSRMM_ARGS(double)),
                        (transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, beta, c, ldc));

/* Used in Intel DAAL via SS */
CALL_RET_FUNC_FROM_DLL(IppStatus, fpk_dft_, ippsSortRadixAscend_64f_I, (Ipp64f * pSrcDst, Ipp64f * pTmp, Ipp32s len), (pSrcDst, pTmp, len));
CALL_RET_FUNC_FROM_DLL(IppStatus, fpk_dft_, ippsSortRadixAscend_32f_I, (Ipp32f * pSrcDst, Ipp32f * pTmp, Ipp32s len), (pSrcDst, pTmp, len));

#define CALL_VOID_FUNC_FROM_DLL_ALONE(fn_dpref, fn_name, argdecl, argcall)                               \
    typedef void(*##fn_dpref##fn_name##_t)##argdecl;                                                     \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr = NULL;                                         \
    void fn_dpref##fn_name##argdecl                                                                      \
    {                                                                                                    \
        load_daal_thr_dll();                                                                             \
        if (##fn_dpref##fn_name##_ptr == NULL)                                                           \
        {                                                                                                \
            ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref #fn_name); \
        }                                                                                                \
        ##fn_dpref##fn_name##_ptr##argcall;                                                              \
    }

#define CALL_RET_FUNC_FROM_DLL_ALONE(ret_type, fn_dpref, fn_name, argdecl, argcall)                      \
    typedef ret_type(*##fn_dpref##fn_name##_t)##argdecl;                                                 \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr = NULL;                                         \
    ret_type fn_dpref##fn_name##argdecl                                                                  \
    {                                                                                                    \
        load_daal_thr_dll();                                                                             \
        if (##fn_dpref##fn_name##_ptr == NULL)                                                           \
        {                                                                                                \
            ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref #fn_name); \
        }                                                                                                \
        return fn_dpref##fn_name##_ptr##argcall;                                                         \
    }

CALL_VOID_FUNC_FROM_DLL_ALONE(fpk_serv_, set_num_threads, (int nth), (nth));
CALL_RET_FUNC_FROM_DLL_ALONE(int, fpk_serv_, get_max_threads, (void), ());
CALL_RET_FUNC_FROM_DLL_ALONE(int, fpk_serv_, set_num_threads_local, (int nth), (nth));
CALL_RET_FUNC_FROM_DLL_ALONE(int, fpk_serv_, get_ncpus, (void), ());
CALL_RET_FUNC_FROM_DLL_ALONE(int, fpk_serv_, get_ncorespercpu, (void), ());
CALL_RET_FUNC_FROM_DLL_ALONE(int, fpk_serv_, get_ht, (void), ());
CALL_RET_FUNC_FROM_DLL_ALONE(int, fpk_serv_, get_nlogicalcores, (void), ());
CALL_RET_FUNC_FROM_DLL_ALONE(int, fpk_serv_, cpuisknm, (void), ());
CALL_RET_FUNC_FROM_DLL_ALONE(int, fpk_serv_, enable_instructions, (int nth), (nth));
CALL_RET_FUNC_FROM_DLL_ALONE(int, fpk_serv_, memmove_s, (void * dest, size_t dmax, const void * src, size_t smax), (dest, dmax, src, smax));

CALL_VOID_FUNC_FROM_DLL_ALONE(fpk_vsl_serv_, threader_for, (DAAL_INT n, DAAL_INT threads_request, void * a, func_type func),
                              (n, threads_request, a, func));
CALL_VOID_FUNC_FROM_DLL_ALONE(fpk_vsl_serv_, threader_for_ordered, (DAAL_INT n, DAAL_INT threads_request, void * a, func_type func),
                              (n, threads_request, a, func));
CALL_VOID_FUNC_FROM_DLL_ALONE(fpk_vsl_serv_, threader_sections, (DAAL_INT threads_request, void * a, func_type func), (threads_request, a, func));
CALL_VOID_FUNC_FROM_DLL_ALONE(fpk_vsl_serv_, threader_ordered, (DAAL_INT i, DAAL_INT th_idx, DAAL_INT th_num, void * a, func_type func),
                              (i, th_idx, th_num, a, func));
CALL_RET_FUNC_FROM_DLL_ALONE(DAAL_INT, fpk_vsl_serv_, threader_get_num_threads_limit, (void), ());
