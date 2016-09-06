/* file: threading.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "daal_defines.h"

namespace daal
{

typedef void (*functype)(int i, const void *a);
typedef void (*functype2)(int i, int n, const void *a);
typedef void *(*tls_functype)(const void *a);
typedef void (*tls_reduce_functype)(void *p, const void *a);

}

extern "C" {

    DAAL_EXPORT int   _daal_threader_get_max_threads();
    DAAL_EXPORT void  _daal_threader_for(int n, int threads_request, const void *a, daal::functype func);
    DAAL_EXPORT void  _daal_threader_for_blocked(int n, int threads_request, const void *a, daal::functype2 func);
    DAAL_EXPORT void *_daal_get_tls_ptr( void *a, daal::tls_functype func );
    DAAL_EXPORT void *_daal_get_tls_local( void *tlsPtr );
    DAAL_EXPORT void  _daal_reduce_tls( void *tlsPtr, void *a, daal::tls_reduce_functype func );
    DAAL_EXPORT void  _daal_del_tls_ptr( void *tlsPtr );
    DAAL_EXPORT bool  _daal_is_in_parallel();

    DAAL_EXPORT size_t _setNumberOfThreads(const size_t numThreads, void **init);
}

namespace daal
{

inline int threader_get_threads_number()
{
    return _daal_threader_get_max_threads();
}

inline size_t setNumberOfThreads(const size_t numThreads, void **init)
{
    return _setNumberOfThreads(numThreads, init);
}

template<typename F>
inline void threader_func(int i, const void *a)
{
    const F &lambda = *static_cast<const F *>(a);
    lambda(i);
}

template<typename F>
inline void threader_func_b(int i0, int in, const void *a)
{
    const F &lambda = *static_cast<const F *>(a);
    lambda(i0, in);
}

template<typename F>
inline void threader_for(int n, int threads_request, const F &lambda)
{
    const void *a = static_cast<const void *>(&lambda);

    _daal_threader_for(n, threads_request, a, threader_func<F>);
}

template<typename F>
inline void threader_for_blocked(int n, int threads_request, const F &lambda)
{
    const void *a = static_cast<const void *>(&lambda);

    _daal_threader_for_blocked(n, threads_request, a, threader_func_b<F>);
}

template<typename lambdaType>
inline void *tls_func(const void *a)
{
    const lambdaType &lambda = *static_cast<const lambdaType *>(a);
    return lambda();
}

template<typename F, typename lambdaType>
inline void tls_reduce_func(void *v, const void *a)
{
    const lambdaType &lambda = *static_cast<const lambdaType *>(a);
    lambda((F)v);
}

struct tlsBase
{
    virtual ~tlsBase() {}
};

class tls_deleter : public tlsBase
{
public:
    virtual void del(void *a) = 0;
};

template<typename lambdaType>
class tls_deleter_ : public tls_deleter
{
public:
    virtual void del(void *a)
    {
        delete static_cast<lambdaType *>(a);
    }
};

template<typename F>
class tls : public tlsBase
{
public:
    template<typename lambdaType>
    explicit tls(const lambdaType &lambda)
    {
        lambdaType *locall = new lambdaType(lambda);
        d = new tls_deleter_<lambdaType>();

        //const void* ac = static_cast<const void*>(&lambda);
        const void *ac = static_cast<const void *>(locall);
        void *a = const_cast<void *>(ac);
        voidLambda = a;

        tlsPtr = _daal_get_tls_ptr( a, tls_func<lambdaType> );
    }

    ~tls()
    {
        d->del(voidLambda);
        delete d;
        _daal_del_tls_ptr( tlsPtr );
    }

    F local()
    {
        void *pf = _daal_get_tls_local( tlsPtr );
        return (static_cast<F>(pf));
    }

    template<typename lambdaType>
    void reduce(const lambdaType &lambda)
    {
        const void *ac = static_cast<const void *>(&lambda);
        void *a = const_cast<void *>(ac);
        _daal_reduce_tls( tlsPtr, a, tls_reduce_func<F, lambdaType> );
    }

private:
    void *tlsPtr;
    void *voidLambda;
    tls_deleter *d;
};

inline bool is_in_parallel()
{
    return _daal_is_in_parallel();
}

}

#endif
