/* file: core_threading_win_dll.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "services/internal/daal_load_win_dynamic_lib.h"
#include "src/threading/threading.h"
#include "src/threading/service_thread_pinner.h"
#include "services/env_detect.h"

static HMODULE daal_thr_dll_handle = NULL;
daal::services::Environment::LibraryThreadingType __daal_serv_get_thr_set();

#define __GLUE__(a, b) a##b

#ifdef _DEBUG
    #define _DLL_SUFFIX(name) __GLUE__(name, "d.2.dll")
#else
    #define _DLL_SUFFIX(name) __GLUE__(name, ".2.dll")
#endif

#define DAAL_LOAD_DLL(name) _daal_load_win_dynamic_lib(name)

DAAL_EXPORT HMODULE load_onedal_thread_dll()
{
    return DAAL_LOAD_DLL(_DLL_SUFFIX("onedal_thread"));
}

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
        daal_thr_dll_handle = load_onedal_thread_dll();
        if (daal_thr_dll_handle == NULL)
        {
            printf("Intel oneDAL FATAL ERROR: Cannot load onedal_thread.2.dll.\n");
            exit(1);
        }
        break;
    }
    default:
    {
        daal_thr_dll_handle = load_onedal_thread_dll();
        if (daal_thr_dll_handle != NULL)
        {
            return;
        }

        printf("Intel oneDAL FATAL ERROR: Cannot load onedal_thread.2.dll.\n");
        exit(1);
    }
    }
}

FARPROC load_daal_thr_func(char * ordinal)
{
    FARPROC FuncAddress;

    if (daal_thr_dll_handle == NULL)
    {
        printf("Intel oneDAL FATAL ERROR: Cannot load \"%s\" function because threaded layer DLL isn`t loaded.\n", ordinal);
        exit(1);
    }

    FuncAddress = GetProcAddress(daal_thr_dll_handle, ordinal);
    if (FuncAddress == NULL)
    {
        printf("GetLastError error code is %lx\n", GetLastError());
        printf("Intel oneDAL FATAL ERROR: Cannot load \"%s\" function.\n", ordinal);
        exit(1);
    }

    return FuncAddress;
}

typedef void * (*_threaded_malloc_t)(const size_t, const size_t);
typedef void (*_threaded_free_t)(void *);

typedef void (*_daal_threader_for_t)(int, int, const void *, daal::functype);
typedef void (*_daal_threader_for_int64_t)(int64_t, const void *, daal::functype_int64);
typedef void (*_daal_threader_for_int32ptr_t)(const int *, const int *, const void *, daal::functype_int32ptr);
typedef void (*_daal_threader_for_simple_t)(int, int, const void *, daal::functype);
typedef void (*_daal_static_threader_for_t)(size_t, const void *, daal::functype_static);
typedef void (*_daal_threader_for_blocked_t)(int, int, const void *, daal::functype2);
typedef void (*_daal_threader_for_blocked_size_t)(size_t, size_t, const void *, daal::functype_blocked_size);
typedef int (*_daal_threader_get_max_threads_t)(void);
typedef int (*_daal_threader_get_current_thread_index_t)(void);
typedef void (*_daal_threader_for_break_t)(int, int, const void *, daal::functype_break);

typedef int64_t (*_daal_parallel_reduce_int32_int64_t)(int32_t, int64_t, const void *, daal::loop_functype_int32_int64, const void *,
                                                       daal::reduction_functype_int64);
typedef int64_t (*_daal_parallel_reduce_int32_int64_t_simple)(int32_t, int64_t, const void *, daal::loop_functype_int32_int64, const void *,
                                                              daal::reduction_functype_int64);
typedef int64_t (*_daal_parallel_reduce_int32ptr_int64_t_simple)(const int32_t *, const int32_t *, int64_t, const void *,
                                                                 daal::loop_functype_int32ptr_int64, const void *, daal::reduction_functype_int64);

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
typedef void (*_daal_tbb_task_scheduler_free_t)(std::shared_ptr<void> globalControl);
typedef size_t (*_setNumberOfThreads_t)(const size_t, std::shared_ptr<void> globalControl, std::shared_ptr<void> scheduleHandle);
typedef void (*_initializeSchedulerHandle_t)(std::shared_ptr<void> scheduleHandle);
typedef void (*_daal_tbb_task_scheduler_handle_finalize_t)(std::shared_ptr<void> globalControl);
typedef void * (*_daal_threader_env_t)();

typedef void (*_daal_parallel_sort_int32_t)(int *, int *);
typedef void (*_daal_parallel_sort_uint64_t)(size_t *, size_t *);
typedef void (*_daal_parallel_sort_pair_int32_uint64_t)(daal::IdxValType<int> *, daal::IdxValType<int> *);
typedef void (*_daal_parallel_sort_pair_fp32_uint64_t)(daal::IdxValType<float> *, daal::IdxValType<float> *);
typedef void (*_daal_parallel_sort_pair_fp64_uint64_t)(daal::IdxValType<double> *, daal::IdxValType<double> *);

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

static _daal_threader_for_t _daal_threader_for_ptr                                           = NULL;
static _daal_threader_for_simple_t _daal_threader_for_simple_ptr                             = NULL;
static _daal_threader_for_int64_t _daal_threader_for_int64_ptr                               = NULL;
static _daal_threader_for_int32ptr_t _daal_threader_for_int32ptr_ptr                         = NULL;
static _daal_static_threader_for_t _daal_static_threader_for_ptr                             = NULL;
static _daal_threader_for_blocked_t _daal_threader_for_blocked_ptr                           = NULL;
static _daal_threader_for_blocked_size_t _daal_threader_for_blocked_size_ptr                 = NULL;
static _daal_threader_for_t _daal_threader_for_optional_ptr                                  = NULL;
static _daal_threader_get_max_threads_t _daal_threader_get_max_threads_ptr                   = NULL;
static _daal_threader_get_current_thread_index_t _daal_threader_get_current_thread_index_ptr = NULL;
static _daal_threader_for_break_t _daal_threader_for_break_ptr                               = NULL;

static _daal_parallel_reduce_int32_int64_t _daal_parallel_reduce_int32_int64_ptr                     = NULL;
static _daal_parallel_reduce_int32_int64_t_simple _daal_parallel_reduce_int32_int64_simple_ptr       = NULL;
static _daal_parallel_reduce_int32ptr_int64_t_simple _daal_parallel_reduce_int32ptr_int64_simple_ptr = NULL;

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

static _daal_is_in_parallel_t _daal_is_in_parallel_ptr                                         = NULL;
static _daal_tbb_task_scheduler_free_t _daal_tbb_task_scheduler_free_ptr                       = NULL;
static _daal_tbb_task_scheduler_handle_finalize_t _daal_tbb_task_scheduler_handle_finalize_ptr = NULL;
static _initializeSchedulerHandle_t _initializeSchedulerHandle_ptr                             = NULL;
static _setNumberOfThreads_t _setNumberOfThreads_ptr                                           = NULL;
static _daal_threader_env_t _daal_threader_env_ptr                                             = NULL;

static _daal_parallel_sort_int32_t _daal_parallel_sort_int32_ptr                         = NULL;
static _daal_parallel_sort_uint64_t _daal_parallel_sort_uint64_ptr                       = NULL;
static _daal_parallel_sort_pair_int32_uint64_t _daal_parallel_sort_pair_int32_uint64_ptr = NULL;
static _daal_parallel_sort_pair_fp32_uint64_t _daal_parallel_sort_pair_fp32_uint64_ptr   = NULL;
static _daal_parallel_sort_pair_fp64_uint64_t _daal_parallel_sort_pair_fp64_uint64_ptr   = NULL;

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

DAAL_EXPORT void _daal_threader_for_simple(int n, int threads_request, const void * a, daal::functype func)
{
    load_daal_thr_dll();
    if (_daal_threader_for_simple_ptr == NULL)
    {
        _daal_threader_for_simple_ptr = (_daal_threader_for_simple_t)load_daal_thr_func("_daal_threader_for_simple");
    }
    _daal_threader_for_simple_ptr(n, threads_request, a, func);
}

DAAL_EXPORT void _daal_threader_for_int32ptr(const int * begin, const int * end, const void * a, daal::functype_int32ptr func)
{
    load_daal_thr_dll();
    if (_daal_threader_for_int32ptr_ptr == NULL)
    {
        _daal_threader_for_int32ptr_ptr = (_daal_threader_for_int32ptr_t)load_daal_thr_func("_daal_threader_for_int32ptr");
    }
    _daal_threader_for_int32ptr_ptr(begin, end, a, func);
}

DAAL_EXPORT void _daal_threader_for_int64(int64_t n, const void * a, daal::functype_int64 func)
{
    load_daal_thr_dll();
    if (_daal_threader_for_int64_ptr == NULL)
    {
        _daal_threader_for_int64_ptr = (_daal_threader_for_int64_t)load_daal_thr_func("_daal_threader_for_int64");
    }
    _daal_threader_for_int64_ptr(n, a, func);
}

DAAL_EXPORT void _daal_static_threader_for(size_t n, const void * a, daal::functype_static func)
{
    load_daal_thr_dll();
    if (_daal_static_threader_for_ptr == NULL)
    {
        _daal_static_threader_for_ptr = (_daal_static_threader_for_t)load_daal_thr_func("_daal_static_threader_for");
    }
    _daal_static_threader_for_ptr(n, a, func);
}

DAAL_EXPORT void _daal_parallel_sort_int32(int * begin_ptr, int * end_ptr)
{
    load_daal_thr_dll();
    if (_daal_parallel_sort_int32_ptr == NULL)
    {
        _daal_parallel_sort_int32_ptr = (_daal_parallel_sort_int32_t)load_daal_thr_func("_daal_parallel_sort_int32");
    }
    _daal_parallel_sort_int32_ptr(begin_ptr, end_ptr);
}

DAAL_EXPORT void _daal_parallel_sort_uint64(size_t * begin_ptr, size_t * end_ptr)
{
    load_daal_thr_dll();
    if (_daal_parallel_sort_uint64_ptr == NULL)
    {
        _daal_parallel_sort_uint64_ptr = (_daal_parallel_sort_uint64_t)load_daal_thr_func("_daal_parallel_sort_uint64");
    }
    _daal_parallel_sort_uint64_ptr(begin_ptr, end_ptr);
}

DAAL_EXPORT void _daal_parallel_sort_pair_int32_uint64(daal::IdxValType<int> * begin_ptr, daal::IdxValType<int> * end_ptr)
{
    load_daal_thr_dll();
    if (_daal_parallel_sort_pair_int32_uint64_ptr == NULL)
    {
        _daal_parallel_sort_pair_int32_uint64_ptr =
            (_daal_parallel_sort_pair_int32_uint64_t)load_daal_thr_func("_daal_parallel_sort_pair_int32_uint64");
    }
    _daal_parallel_sort_pair_int32_uint64_ptr(begin_ptr, end_ptr);
}

DAAL_EXPORT void _daal_parallel_sort_pair_fp32_uint64(daal::IdxValType<float> * begin_ptr, daal::IdxValType<float> * end_ptr)
{
    load_daal_thr_dll();
    if (_daal_parallel_sort_pair_fp32_uint64_ptr == NULL)
    {
        _daal_parallel_sort_pair_fp32_uint64_ptr = (_daal_parallel_sort_pair_fp32_uint64_t)load_daal_thr_func("_daal_parallel_sort_pair_fp32_uint64");
    }
    _daal_parallel_sort_pair_fp32_uint64_ptr(begin_ptr, end_ptr);
}

DAAL_EXPORT void _initializeSchedulerHandle(std::shared_ptr<void> init)
{
    load_daal_thr_dll();
    if (_initializeSchedulerHandle_ptr == NULL)
    {
        _initializeSchedulerHandle_ptr = (_initializeSchedulerHandle_t)load_daal_thr_func("_initializeSchedulerHandle");
    }
    _initializeSchedulerHandle_ptr(init);
}

DAAL_EXPORT void _daal_tbb_task_scheduler_handle_finalize(std::shared_ptr<void> init)
{
    if (init == NULL)
    {
        // If threading library was not opened, there is nothing to free,
        // so we do not need to load threading library.
        // Moreover, loading threading library in the Environment destructor
        // results in a crush because of the use of Wintrust library after it was unloaded.
        // This happens due to undefined order of static objects deinitialization
        // like Environment, and dependent libraries.
        return;
    }
    load_daal_thr_dll();
    if (_daal_tbb_task_scheduler_handle_finalize_ptr == NULL)
    {
        _daal_tbb_task_scheduler_handle_finalize_ptr =
            (_daal_tbb_task_scheduler_handle_finalize_t)load_daal_thr_func("_daal_tbb_task_scheduler_handle_finalize");
    }
    _daal_tbb_task_scheduler_handle_finalize_ptr(init);
}

DAAL_EXPORT void _daal_parallel_sort_pair_fp64_uint64(daal::IdxValType<double> * begin_ptr, daal::IdxValType<double> * end_ptr)
{
    load_daal_thr_dll();
    if (_daal_parallel_sort_pair_fp64_uint64_ptr == NULL)
    {
        _daal_parallel_sort_pair_fp64_uint64_ptr = (_daal_parallel_sort_pair_fp64_uint64_t)load_daal_thr_func("_daal_parallel_sort_pair_fp64_uint64");
    }
    _daal_parallel_sort_pair_fp64_uint64_ptr(begin_ptr, end_ptr);
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

DAAL_EXPORT void _daal_threader_for_blocked_size(size_t n, size_t block, const void * a, daal::functype_blocked_size func)
{
    load_daal_thr_dll();
    if (_daal_threader_for_blocked_size_ptr == NULL)
    {
        _daal_threader_for_blocked_size_ptr = (_daal_threader_for_blocked_size_t)load_daal_thr_func("_daal_threader_for_blocked_size");
    }
    _daal_threader_for_blocked_size_ptr(n, block, a, func);
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

DAAL_EXPORT int64_t _daal_parallel_reduce_int32_int64(int32_t n, int64_t init, const void * a, daal::loop_functype_int32_int64 loop_func,
                                                      const void * b, daal::reduction_functype_int64 reduction_func)
{
    load_daal_thr_dll();
    if (_daal_parallel_reduce_int32_int64_ptr == NULL)
    {
        _daal_parallel_reduce_int32_int64_ptr = (_daal_parallel_reduce_int32_int64_t)load_daal_thr_func("_daal_parallel_reduce_int32_int64");
    }
    return _daal_parallel_reduce_int32_int64_ptr(n, init, a, loop_func, b, reduction_func);
}

DAAL_EXPORT int64_t _daal_parallel_reduce_int32_int64_simple(int32_t n, int64_t init, const void * a, daal::loop_functype_int32_int64 loop_func,
                                                             const void * b, daal::reduction_functype_int64 reduction_func)
{
    load_daal_thr_dll();
    if (_daal_parallel_reduce_int32_int64_simple_ptr == NULL)
    {
        _daal_parallel_reduce_int32_int64_simple_ptr =
            (_daal_parallel_reduce_int32_int64_t_simple)load_daal_thr_func("_daal_parallel_reduce_int32_int64_simple");
    }
    return _daal_parallel_reduce_int32_int64_simple_ptr(n, init, a, loop_func, b, reduction_func);
}

DAAL_EXPORT int64_t _daal_parallel_reduce_int32ptr_int64_simple(const int32_t * begin, const int32_t * end, int64_t init, const void * a,
                                                                daal::loop_functype_int32ptr_int64 loop_func, const void * b,
                                                                daal::reduction_functype_int64 reduction_func)
{
    load_daal_thr_dll();
    if (_daal_parallel_reduce_int32ptr_int64_simple_ptr == NULL)
    {
        _daal_parallel_reduce_int32ptr_int64_simple_ptr =
            (_daal_parallel_reduce_int32ptr_int64_t_simple)load_daal_thr_func("_daal_parallel_reduce_int32ptr_int64_simple");
    }
    return _daal_parallel_reduce_int32ptr_int64_simple_ptr(begin, end, init, a, loop_func, b, reduction_func);
}

DAAL_EXPORT void _daal_threader_for_break(int n, int threads_request, const void * a, daal::functype_break func)
{
    load_daal_thr_dll();
    if (_daal_threader_for_break_ptr == NULL)
    {
        _daal_threader_for_break_ptr = (_daal_threader_for_break_t)load_daal_thr_func("_daal_threader_for_break");
    }
    _daal_threader_for_break_ptr(n, threads_request, a, func);
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

DAAL_EXPORT int _daal_threader_get_current_thread_index()
{
    load_daal_thr_dll();
    if (_daal_threader_get_current_thread_index_ptr == NULL)
    {
        _daal_threader_get_current_thread_index_ptr =
            (_daal_threader_get_current_thread_index_t)load_daal_thr_func("_daal_threader_get_current_thread_index");
    }
    return _daal_threader_get_current_thread_index_ptr();
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

DAAL_EXPORT void _daal_tbb_task_scheduler_free(std::shared_ptr<void> init)
{
    load_daal_thr_dll();
    if (_daal_tbb_task_scheduler_free_ptr == NULL)
    {
        _daal_tbb_task_scheduler_free_ptr = (_daal_tbb_task_scheduler_free_t)load_daal_thr_func("_daal_tbb_task_scheduler_free");
    }
    return _daal_tbb_task_scheduler_free_ptr(init);
}

DAAL_EXPORT size_t _setNumberOfThreads(const size_t numThreads, std::shared_ptr<void> init, std::shared_ptr<void> control)
{
    load_daal_thr_dll();
    if (_setNumberOfThreads_ptr == NULL)
    {
        _setNumberOfThreads_ptr = (_setNumberOfThreads_t)load_daal_thr_func("_setNumberOfThreads");
    }
    return _setNumberOfThreads_ptr(numThreads, init, control);
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

#define CALL_VOID_FUNC_FROM_DLL(fn_dpref, fn_name, argdecl, argcall)          \
    typedef void(*##fn_dpref##fn_name##_t)##argdecl;                          \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr = NULL;              \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, avx512_, fn_name, argdecl, argcall) \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, avx2_, fn_name, argdecl, argcall)   \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, sse42_, fn_name, argdecl, argcall)  \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, sse2_, fn_name, argdecl, argcall)

#define CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref, fn_cpu, fn_name, argdecl, argcall)                                 \
    extern "C" DAAL_EXPORT void fn_dpref##fn_cpu##fn_name##argdecl                                               \
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
        extern "C" DAAL_EXPORT void fn_dpref##fn_cpu##fn_name##argdecl                                               \
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

#define CALL_RET_FUNC_FROM_DLL(ret_type, fn_dpref, fn_name, argdecl, argcall)          \
    typedef ret_type(*##fn_dpref##fn_name##_t)##argdecl;                               \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr = NULL;                       \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, avx512_, fn_name, argdecl, argcall) \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, avx2_, fn_name, argdecl, argcall)   \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, sse42_, fn_name, argdecl, argcall)  \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, sse2_, fn_name, argdecl, argcall)

#define CALL_RET_FUNC_FROM_DLL_CPU(ret_type, fn_dpref, fn_cpu, fn_name, argdecl, argcall)                        \
    extern "C" DAAL_EXPORT ret_type fn_dpref##fn_cpu##fn_name##argdecl                                           \
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
        extern "C" DAAL_EXPORT ret_type fn_dpref##fn_cpu##fn_name##argdecl                                           \
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

/* Used directly in Intel(R) oneAPI Data Analytics Library (oneDAL) */
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

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgetrf,
                        (const DAAL_INT * m, const DAAL_INT * n, const double * a, const DAAL_INT * lda, const DAAL_INT * ipiv, DAAL_INT * info),
                        (m, n, a, lda, ipiv, info));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgetrf,
                        (const DAAL_INT * m, const DAAL_INT * n, const float * a, const DAAL_INT * lda, const DAAL_INT * ipiv, DAAL_INT * info),
                        (m, n, a, lda, ipiv, info));

CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgetrs,
                        (const char * trans, const DAAL_INT * n, const DAAL_INT * nrhs, const double * a, const DAAL_INT * lda, const DAAL_INT * ipiv,
                         double * b, const DAAL_INT * ldb, DAAL_INT * info, int iuplo),
                        (trans, n, nrhs, a, lda, ipiv, b, ldb, info, iuplo));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgetrs,
                        (const char * trans, const DAAL_INT * n, const DAAL_INT * nrhs, const float * a, const DAAL_INT * lda, const DAAL_INT * ipiv,
                         float * b, const DAAL_INT * ldb, DAAL_INT * info, int iuplo),
                        (trans, n, nrhs, a, lda, ipiv, b, ldb, info, iuplo));

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

typedef int IppStatus;
typedef unsigned char Ipp8u;
typedef unsigned short Ipp16u;
typedef unsigned int Ipp32u;
typedef signed short Ipp16s;
typedef signed int Ipp32s;
typedef float Ipp32f;
typedef double Ipp64f;

/* Used in Intel(R) oneAPI Data Analytics Library (oneDAL) via SS */
CALL_RET_FUNC_FROM_DLL(IppStatus, fpk_dft_, ippsSortRadixAscend_64f_I, (Ipp64f * pSrcDst, Ipp64f * pTmp, Ipp32s len), (pSrcDst, pTmp, len));
CALL_RET_FUNC_FROM_DLL(IppStatus, fpk_dft_, ippsSortRadixAscend_32f_I, (Ipp32f * pSrcDst, Ipp32f * pTmp, Ipp32s len), (pSrcDst, pTmp, len));

#define CALL_VOID_FUNC_FROM_DLL_ALONE(fn_dpref, fn_name, argdecl, argcall)                               \
    typedef void(*##fn_dpref##fn_name##_t)##argdecl;                                                     \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr = NULL;                                         \
    extern "C" DAAL_EXPORT void fn_dpref##fn_name##argdecl                                               \
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
    extern "C" DAAL_EXPORT ret_type fn_dpref##fn_name##argdecl                                           \
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

typedef void (*func_type)(DAAL_INT, DAAL_INT, DAAL_INT, void *);

CALL_VOID_FUNC_FROM_DLL_ALONE(fpk_vsl_serv_, threader_for, (DAAL_INT n, DAAL_INT threads_request, void * a, func_type func),
                              (n, threads_request, a, func));
CALL_VOID_FUNC_FROM_DLL_ALONE(fpk_vsl_serv_, threader_for_ordered, (DAAL_INT n, DAAL_INT threads_request, void * a, func_type func),
                              (n, threads_request, a, func));
CALL_VOID_FUNC_FROM_DLL_ALONE(fpk_vsl_serv_, threader_sections, (DAAL_INT threads_request, void * a, func_type func), (threads_request, a, func));
CALL_VOID_FUNC_FROM_DLL_ALONE(fpk_vsl_serv_, threader_ordered, (DAAL_INT i, DAAL_INT th_idx, DAAL_INT th_num, void * a, func_type func),
                              (i, th_idx, th_num, a, func));
CALL_RET_FUNC_FROM_DLL_ALONE(DAAL_INT, fpk_vsl_serv_, threader_get_num_threads_limit, (void), ());
