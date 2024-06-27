/* file: threading.h */
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
//  Declaration of threading layer functions.
//--
*/

#ifndef __THREADING_H__
#define __THREADING_H__

#include <stdint.h>
#include "services/daal_defines.h"

namespace daal
{
template <typename FPType>
struct IdxValType
{
    FPType value;
    size_t index;

    bool operator<(const IdxValType & o) const { return o.value == value ? index < o.index : value < o.value; }
    bool operator>(const IdxValType & o) const { return o.value == value ? index > o.index : value > o.value; }
    bool operator<=(const IdxValType & o) const { return value < o.value || (value == o.value && index == o.index); }
};
typedef void (*functype)(int i, const void * a);
typedef void (*functype_int64)(int64_t i, const void * a);
typedef void (*functype_int32ptr)(const int * i, const void * a);
typedef void (*functype_static)(size_t i, size_t tid, const void * a);
typedef void (*functype2)(int i, int n, const void * a);
typedef void (*functype_blocked_size)(size_t first, size_t last, const void * a);
typedef void * (*tls_functype)(const void * a);
typedef void (*tls_reduce_functype)(void * p, const void * a);
typedef void (*functype_break)(int i, bool & needBreak, const void * a);
typedef int64_t (*loop_functype_int32_int64)(int32_t start_idx_reduce, int32_t end_idx_reduce, int64_t value_for_reduce, const void * a);
typedef int64_t (*loop_functype_int32ptr_int64)(const int32_t * start_idx_reduce, const int32_t * end_idx_reduce, int64_t value_for_reduce,
                                                const void * a);
typedef int64_t (*reduction_functype_int64)(int64_t a, int64_t b, const void * reduction);

class task;
} // namespace daal

extern "C"
{
    DAAL_EXPORT int _daal_threader_get_max_threads();
    DAAL_EXPORT int _daal_threader_get_current_thread_index();
    DAAL_EXPORT void _daal_threader_for(int n, int threads_request, const void * a, daal::functype func);
    DAAL_EXPORT void _daal_threader_for_int64(int64_t n, const void * a, daal::functype_int64 func);
    DAAL_EXPORT void _daal_threader_for_simple(int n, int threads_request, const void * a, daal::functype func);
    DAAL_EXPORT void _daal_threader_for_int32ptr(const int * begin, const int * end, const void * a, daal::functype_int32ptr func);
    DAAL_EXPORT void _daal_static_threader_for(size_t n, const void * a, daal::functype_static func);
    DAAL_EXPORT void _daal_threader_for_blocked(int n, int threads_request, const void * a, daal::functype2 func);
    DAAL_EXPORT void _daal_threader_for_blocked_size(size_t n, size_t block, const void * a, daal::functype_blocked_size func);
    DAAL_EXPORT void _daal_threader_for_optional(int n, int threads_request, const void * a, daal::functype func);
    DAAL_EXPORT void _daal_threader_for_break(int n, int threads_request, const void * a, daal::functype_break func);

    DAAL_EXPORT int64_t _daal_parallel_reduce_int32_int64(int32_t n, int64_t init, const void * a, daal::loop_functype_int32_int64 loop_func,
                                                          const void * b, daal::reduction_functype_int64 reduction_func);
    DAAL_EXPORT int64_t _daal_parallel_reduce_int32_int64_simple(int32_t n, int64_t init, const void * a, daal::loop_functype_int32_int64 loop_func,
                                                                 const void * b, daal::reduction_functype_int64 reduction_func);
    DAAL_EXPORT int64_t _daal_parallel_reduce_int32ptr_int64_simple(const int32_t * begin, const int32_t * end, int64_t init, const void * a,
                                                                    daal::loop_functype_int32ptr_int64 loop_func, const void * b,
                                                                    daal::reduction_functype_int64 reduction_func);

    DAAL_EXPORT void * _daal_get_tls_ptr(void * a, daal::tls_functype func);
    DAAL_EXPORT void * _daal_get_tls_local(void * tlsPtr);
    DAAL_EXPORT void _daal_reduce_tls(void * tlsPtr, void * a, daal::tls_reduce_functype func);
    DAAL_EXPORT void _daal_parallel_reduce_tls(void * tlsPtr, void * a, daal::tls_reduce_functype func);
    DAAL_EXPORT void _daal_del_tls_ptr(void * tlsPtr);

    DAAL_EXPORT void * _daal_get_ls_ptr(void * a, daal::tls_functype func);
    DAAL_EXPORT void * _daal_get_ls_local(void * lsPtr);
    DAAL_EXPORT void _daal_release_ls_local(void * lsPtr, void * p);
    DAAL_EXPORT void _daal_reduce_ls(void * lsPtr, void * a, daal::tls_reduce_functype func);
    DAAL_EXPORT void _daal_del_ls_ptr(void * lsPtr);

    DAAL_EXPORT void * _daal_new_mutex();
    DAAL_EXPORT void _daal_lock_mutex(void * mutexPtr);
    DAAL_EXPORT void _daal_unlock_mutex(void * mutexPtr);
    DAAL_EXPORT void _daal_del_mutex(void * mutexPtr);
    DAAL_EXPORT bool _daal_is_in_parallel();

    DAAL_EXPORT void * _daal_new_task_group();
    DAAL_EXPORT void _daal_del_task_group(void * taskGroupPtr);
    DAAL_EXPORT void _daal_run_task_group(void * taskGroupPtr, daal::task * t);
    DAAL_EXPORT void _daal_wait_task_group(void * taskGroupPtr);

    DAAL_EXPORT void _daal_tbb_task_arena_free(void *& taskArena);
    DAAL_EXPORT void _daal_tbb_task_scheduler_free(void *& globalControl);
    DAAL_EXPORT size_t _setNumberOfThreads(const size_t numThreads);

    DAAL_EXPORT void * _daal_threader_env();

    DAAL_EXPORT void * _threaded_scalable_malloc(const size_t size, const size_t alignment);
    DAAL_EXPORT void _threaded_scalable_free(void * ptr);

#define DAAL_PARALLEL_SORT_DECL(TYPE, NAMESUFFIX) DAAL_EXPORT void _daal_parallel_sort_##NAMESUFFIX(TYPE * begin_ptr, TYPE * end_ptr);
    DAAL_PARALLEL_SORT_DECL(int, int32)
    DAAL_PARALLEL_SORT_DECL(size_t, uint64)
    DAAL_PARALLEL_SORT_DECL(daal::IdxValType<int>, pair_int32_uint64)
    DAAL_PARALLEL_SORT_DECL(daal::IdxValType<float>, pair_fp32_uint64)
    DAAL_PARALLEL_SORT_DECL(daal::IdxValType<double>, pair_fp64_uint64)
#undef DAAL_PARALLEL_SORT_DECL
}

namespace daal
{
template <typename FPType>
inline void parallel_sort(daal::IdxValType<FPType> * beginPtr, daal::IdxValType<FPType> * endPtr)
{}

template <>
inline void parallel_sort<int>(daal::IdxValType<int> * beginPtr, daal::IdxValType<int> * endPtr)
{
    _daal_parallel_sort_pair_int32_uint64(beginPtr, endPtr);
}

template <>
inline void parallel_sort<float>(daal::IdxValType<float> * beginPtr, daal::IdxValType<float> * endPtr)
{
    _daal_parallel_sort_pair_fp32_uint64(beginPtr, endPtr);
}

template <>
inline void parallel_sort<double>(daal::IdxValType<double> * beginPtr, daal::IdxValType<double> * endPtr)
{
    _daal_parallel_sort_pair_fp64_uint64(beginPtr, endPtr);
}

inline int threader_get_max_threads_number()
{
    return _daal_threader_get_max_threads();
}

inline int threader_get_max_current_thread_index()
{
    return _daal_threader_get_current_thread_index();
}

inline void * threaded_scalable_malloc(const size_t size, const size_t alignment)
{
    return _threaded_scalable_malloc(size, alignment);
}

inline void threaded_scalable_free(void * ptr)
{
    _threaded_scalable_free(ptr);
}

class ThreaderEnvironment
{
public:
    ThreaderEnvironment() : _numberOfThreads(1 /*_daal_threader_get_max_threads()*/) {}
    ~ThreaderEnvironment() { _daal_tbb_task_arena_free(_taskArena); }
    size_t getNumberOfThreads() const { return _numberOfThreads; }
    void setNumberOfThreads(size_t value) { _numberOfThreads = value; }
    void *& getTaskArena() { return _taskArena; }

private:
    size_t _numberOfThreads;
    void * _taskArena;
};

inline ThreaderEnvironment * threader_env()
{
    return static_cast<ThreaderEnvironment *>(_daal_threader_env());
}

inline size_t threader_get_threads_number()
{
    return threader_env()->getNumberOfThreads();
}

inline size_t setNumberOfThreads(const size_t numThreads)
{
    return _setNumberOfThreads(numThreads);
}

template <typename F>
inline void threader_func(int i, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(i);
}

template <typename F>
inline void static_threader_func(size_t i, size_t tid, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(i, tid);
}

template <typename F>
inline void threader_func_b(int i0, int in, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(i0, in);
}

template <typename F>
inline void threader_func_break(int i, bool & needBreak, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(i, needBreak);
}

template <typename F>
inline void threader_for(int n, int threads_request, const F & lambda)
{
    const void * a = static_cast<const void *>(&lambda);

    _daal_threader_for(n, threads_request, a, threader_func<F>);
}

template <typename F>
inline void threader_for_int64(int64_t n, const F & lambda)
{
    const void * a = static_cast<const void *>(&lambda);

    _daal_threader_for_int64(n, a, threader_func<F>);
}

template <typename F>
inline void threader_for_simple(int n, int threads_request, const F & lambda)
{
    const void * a = static_cast<const void *>(&lambda);

    _daal_threader_for_simple(n, threads_request, a, threader_func<F>);
}

template <typename F>
inline void threader_for_int32ptr(const int * begin, const int * end, const F & lambda)
{
    const void * a = static_cast<const void *>(&lambda);

    _daal_threader_for_int32ptr(begin, end, a, threader_func<F>);
}

template <typename F>
inline void static_threader_for(size_t n, const F & lambda)
{
    const void * a = static_cast<const void *>(&lambda);

    _daal_static_threader_for(n, a, static_threader_func<F>);
}

template <typename F>
inline void threader_for_blocked(int n, int threads_request, const F & lambda)
{
    const void * a = static_cast<const void *>(&lambda);

    _daal_threader_for_blocked(n, threads_request, a, threader_func_b<F>);
}

template <typename F>
inline void threader_for_optional(int n, int threads_request, const F & lambda)
{
    const void * a = static_cast<const void *>(&lambda);

    _daal_threader_for_optional(n, threads_request, a, threader_func<F>);
}

template <typename F>
inline void threader_for_break(int n, int threads_request, const F & lambda)
{
    const void * a = static_cast<const void *>(&lambda);

    _daal_threader_for_break(n, threads_request, a, threader_func_break<F>);
}

template <typename lambdaType>
inline void * tls_func(const void * a)
{
    const lambdaType & lambda = *static_cast<const lambdaType *>(a);
    return lambda();
}

template <typename F, typename lambdaType>
inline void tls_reduce_func(void * v, const void * a)
{
    const lambdaType & lambda = *static_cast<const lambdaType *>(a);
    lambda((F)v);
}

struct tlsBase
{
    virtual ~tlsBase() {}
};

class tls_deleter : public tlsBase
{
public:
    virtual ~tls_deleter() {}
    virtual void del(void * a) = 0;
};

template <typename lambdaType>
class tls_deleter_ : public tls_deleter
{
public:
    virtual ~tls_deleter_() {}
    virtual void del(void * a) { delete static_cast<lambdaType *>(a); }
};

template <typename F>
class tls : public tlsBase
{
public:
    template <typename lambdaType>
    explicit tls(const lambdaType & lambda)
    {
        lambdaType * locall = new lambdaType(lambda);
        d                   = new tls_deleter_<lambdaType>();

        //const void* ac = static_cast<const void*>(&lambda);
        const void * ac = static_cast<const void *>(locall);
        void * a        = const_cast<void *>(ac);
        voidLambda      = a;

        tlsPtr = _daal_get_tls_ptr(a, tls_func<lambdaType>);
    }

    virtual ~tls()
    {
        d->del(voidLambda);
        delete d;
        _daal_del_tls_ptr(tlsPtr);
    }

    F local()
    {
        void * pf = _daal_get_tls_local(tlsPtr);
        return (static_cast<F>(pf));
    }

    template <typename lambdaType>
    void reduce(const lambdaType & lambda)
    {
        const void * ac = static_cast<const void *>(&lambda);
        void * a        = const_cast<void *>(ac);
        _daal_reduce_tls(tlsPtr, a, tls_reduce_func<F, lambdaType>);
    }

    template <typename lambdaType>
    void parallel_reduce(const lambdaType & lambda)
    {
        const void * ac = static_cast<const void *>(&lambda);
        void * a        = const_cast<void *>(ac);
        _daal_parallel_reduce_tls(tlsPtr, a, tls_reduce_func<F, lambdaType>);
    }

private:
    void * tlsPtr;
    void * voidLambda;
    tls_deleter * d;
};

template <typename F, typename lambdaType>
inline void * creater_func(const void * a)
{
    const lambdaType & lambda = *static_cast<const lambdaType *>(a);
    return lambda();
}

class static_tls_deleter
{
public:
    virtual ~static_tls_deleter() {}
    virtual void del(void * a) = 0;
};

template <typename lambdaType>
class static_tls_deleter_ : public static_tls_deleter
{
public:
    virtual ~static_tls_deleter_() {}
    virtual void del(void * a) { delete static_cast<lambdaType *>(a); }
};

template <typename F>
class static_tls
{
public:
    template <typename lambdaType>
    explicit static_tls(const lambdaType & lambda)
    {
        _nThreads = threader_get_max_threads_number();

        _storage = new F[_nThreads];

        if (!_storage)
        {
            return;
        }

        for (size_t i = 0; i < _nThreads; ++i)
        {
            _storage[i] = nullptr;
        }

        lambdaType * locall = new lambdaType(lambda);
        _deleter            = new static_tls_deleter_<lambdaType>();
        if (!locall || !_deleter)
        {
            return;
        }

        const void * ac = static_cast<const void *>(locall);
        void * a        = const_cast<void *>(ac);
        _creater        = a;

        _creater_func = creater_func<F, lambdaType>;
    }

    virtual ~static_tls()
    {
        if (_deleter)
        {
            _deleter->del(_creater);
        }
        delete _deleter;
        delete[] _storage;
    }

    F local(size_t tid)
    {
        if (_storage)
        {
            if (!_storage[tid])
            {
                _storage[tid] = static_cast<F>(_creater_func(_creater));
            }

            return _storage[tid];
        }
        else
        {
            return nullptr;
        }
    }

    template <typename lambdaType>
    void reduce(const lambdaType & lambda)
    {
        if (_storage)
        {
            for (size_t i = 0; i < _nThreads; ++i)
            {
                if (_storage[i]) lambda(_storage[i]);
            }
        }
    }

    size_t nthreads() const { return _nThreads; }

private:
    F * _storage                     = nullptr;
    size_t _nThreads                 = 0;
    void * _creater                  = nullptr;
    daal::tls_functype _creater_func = nullptr;
    static_tls_deleter * _deleter    = nullptr;
};

template <typename F>
class ls : public tlsBase
{
public:
    template <typename lambdaType>
    explicit ls(const lambdaType & lambda, const bool isTls = false)
    {
        _isTls              = isTls;
        lambdaType * locall = new lambdaType(lambda);
        d                   = new tls_deleter_<lambdaType>();

        //const void* ac = static_cast<const void*>(&lambda);
        const void * ac = static_cast<const void *>(locall);
        void * a        = const_cast<void *>(ac);
        voidLambda      = a;

        lsPtr = _isTls ? _daal_get_tls_ptr(a, tls_func<lambdaType>) : _daal_get_ls_ptr(a, tls_func<lambdaType>);
    }

    virtual ~ls()
    {
        d->del(voidLambda);
        delete d;
        _isTls ? _daal_del_tls_ptr(lsPtr) : _daal_del_ls_ptr(lsPtr);
    }

    F local()
    {
        void * pf = _isTls ? _daal_get_tls_local(lsPtr) : _daal_get_ls_local(lsPtr);
        return (static_cast<F>(pf));
    }

    void release(F p)
    {
        if (!_isTls) _daal_release_ls_local(lsPtr, p);
    }

    template <typename lambdaType>
    void reduce(const lambdaType & lambda)
    {
        const void * ac = static_cast<const void *>(&lambda);
        void * a        = const_cast<void *>(ac);
        _isTls ? _daal_reduce_tls(lsPtr, a, tls_reduce_func<F, lambdaType>) : _daal_reduce_ls(lsPtr, a, tls_reduce_func<F, lambdaType>);
    }

private:
    void * lsPtr;
    void * voidLambda;
    tls_deleter * d;
    bool _isTls;
};

template <typename F>
class ls_release
{
public:
    ls_release(ls<F *> & t, F * p) : _t(t), _p(p) {}
    ~ls_release() { _t.release(_p); }

private:
    ls<F *> & _t;
    F * _p;
};
#define DAAL_LS_RELEASE(T, t, p) ls_release<T> __auto_release(t, p);

class DAAL_EXPORT task
{
public:
    virtual void run()     = 0;
    virtual void destroy() = 0;

protected:
    task() {}
    virtual ~task() {}
};

inline bool is_in_parallel()
{
    return _daal_is_in_parallel();
}

template <typename Func>
void conditional_threader_for(const bool inParallel, const size_t n, Func func)
{
    if (inParallel)
    {
        threader_for(n, n, [&](size_t i) { func(i); });
    }
    else
    {
        for (size_t i = 0; i < n; ++i)
        {
            func(i);
        }
    }
}

template <typename Func>
void conditional_static_threader_for(const bool inParallel, const size_t n, Func func)
{
    if (inParallel)
    {
        static_threader_for(n, [&](size_t i, size_t tid) { func(i, tid); });
    }
    else
    {
        for (size_t i = 0; i < n; ++i)
        {
            func(i, 0);
        }
    }
}

} // namespace daal

#endif
