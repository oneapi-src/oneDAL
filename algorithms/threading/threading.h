/* file: threading.h */
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
//  Declaration of threading layer functions.
//--
*/

#ifndef __THREADING_H__
#define __THREADING_H__

#include "services/daal_defines.h"

namespace daal
{
typedef void (*functype)(int i, const void * a);
typedef void (*functype2)(int i, int n, const void * a);
typedef void * (*tls_functype)(const void * a);
typedef void (*tls_reduce_functype)(void * p, const void * a);
typedef void (*init_functype)(void ** value, const void * f);
typedef void (*delete_functype)(void ** value, const void * f);
typedef void (*loop_functype)(void * local, size_t begin, size_t end, const void * f);
typedef void (*reduce_functype)(void * lhs, void * rhs, const void * f);
class task;
} // namespace daal

extern "C"
{
    DAAL_EXPORT int _daal_threader_get_max_threads();
    DAAL_EXPORT void _daal_threader_for(int n, int threads_request, const void * a, daal::functype func);
    DAAL_EXPORT void _daal_threader_for_blocked(int n, int threads_request, const void * a, daal::functype2 func);
    DAAL_EXPORT void _daal_threader_for_optional(int n, int threads_request, const void * a, daal::functype func);

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

    DAAL_EXPORT void _daal_tbb_task_scheduler_free(void *& init);
    DAAL_EXPORT size_t _setNumberOfThreads(const size_t numThreads, void ** init);

    DAAL_EXPORT void * _daal_threader_env();

    DAAL_EXPORT void * _threaded_scalable_malloc(const size_t size, const size_t alignment);
    DAAL_EXPORT void _threaded_scalable_free(void * ptr);

    DAAL_EXPORT void * _daal_parallel_deterministic_reduce(size_t n, size_t grain_size, const void * a, const void * b, const void * c,
                                                           const void * d, daal::init_functype init_func, daal::delete_functype delete_func,
                                                           daal::loop_functype loop_func, daal::reduce_functype reduce_func);
}

namespace daal
{
inline int threader_get_max_threads_number()
{
    return _daal_threader_get_max_threads();
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
    ThreaderEnvironment() : _numberOfThreads(_daal_threader_get_max_threads()) {}
    size_t getNumberOfThreads() const { return _numberOfThreads; }
    void setNumberOfThreads(size_t value) { _numberOfThreads = value; }

private:
    size_t _numberOfThreads;
};

inline ThreaderEnvironment * threader_env()
{
    return static_cast<ThreaderEnvironment *>(_daal_threader_env());
}

inline size_t threader_get_threads_number()
{
    return threader_env()->getNumberOfThreads();
}

inline size_t setNumberOfThreads(const size_t numThreads, void ** init)
{
    return _setNumberOfThreads(numThreads, init);
}

template <typename F>
inline void threader_func(int i, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(i);
}

template <typename F>
inline void threader_func_b(int i0, int in, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(i0, in);
}

template <typename F>
inline void threader_for(int n, int threads_request, const F & lambda)
{
    const void * a = static_cast<const void *>(&lambda);

    _daal_threader_for(n, threads_request, a, threader_func<F>);
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

template <typename F>
class ls : public tlsBase
{
public:
    template <typename lambdaType>
    explicit ls(const lambdaType & lambda)
    {
        lambdaType * locall = new lambdaType(lambda);
        d                   = new tls_deleter_<lambdaType>();

        //const void* ac = static_cast<const void*>(&lambda);
        const void * ac = static_cast<const void *>(locall);
        void * a        = const_cast<void *>(ac);
        voidLambda      = a;

        lsPtr = _daal_get_ls_ptr(a, tls_func<lambdaType>);
    }

    virtual ~ls()
    {
        d->del(voidLambda);
        delete d;
        _daal_del_ls_ptr(lsPtr);
    }

    F local()
    {
        void * pf = _daal_get_ls_local(lsPtr);
        return (static_cast<F>(pf));
    }

    void release(F p) { _daal_release_ls_local(lsPtr, p); }

    template <typename lambdaType>
    void reduce(const lambdaType & lambda)
    {
        const void * ac = static_cast<const void *>(&lambda);
        void * a        = const_cast<void *>(ac);
        _daal_reduce_ls(lsPtr, a, tls_reduce_func<F, lambdaType>);
    }

private:
    void * lsPtr;
    void * voidLambda;
    tls_deleter * d;
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

template <typename F>
inline void init_func(void ** value, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(value);
}

template <typename F>
inline void delete_func(void ** value, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(value);
}

template <typename F>
inline void loop_func(void * local, size_t begin, size_t end, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(local, begin, end);
}

template <typename F>
inline void reduce_func(void * lhs, void * rhs, const void * a)
{
    const F & lambda = *static_cast<const F *>(a);
    lambda(lhs, rhs);
}

template <typename F1, typename F2, typename F3, typename F4>
inline void * parallel_deterministic_reduce(size_t n, size_t grain_size, const F1 & init_function, const F2 & delete_function,
                                            const F3 & loop_function, const F4 & reduce_function)
{
    const void * a = static_cast<const void *>(&init_function);
    const void * b = static_cast<const void *>(&delete_function);
    const void * c = static_cast<const void *>(&loop_function);
    const void * d = static_cast<const void *>(&reduce_function);

    return _daal_parallel_deterministic_reduce(n, grain_size, a, b, c, d, init_func<F1>, delete_func<F2>, loop_func<F3>, reduce_func<F4>);
}

} // namespace daal

#endif
