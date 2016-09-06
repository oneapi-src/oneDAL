/* file: threading.cpp */
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
//  Implementation of threading layer functions.
//--
*/

#include "threading.h"

#if defined(__DO_TBB_LAYER__)
    #include <tbb/tbb.h>
    #include <tbb/spin_mutex.h>
#endif

DAAL_EXPORT size_t _setNumberOfThreads(const size_t numThreads, void** init)
{
  #if defined(__DO_TBB_LAYER__)
    static tbb::spin_mutex mt;
    tbb::spin_mutex::scoped_lock lock(mt);
    if(numThreads != 0)
    {
        if(*init)
        {
            delete ((tbb::task_scheduler_init *)(*init));
        }
        *init = (void *)(new tbb::task_scheduler_init(numThreads));
        return numThreads;
    }
  #endif
    return 1;
}

DAAL_EXPORT void _daal_threader_for(int n, int threads_request, const void* a, daal::functype func)
{
  #if defined(__DO_TBB_LAYER__)
    tbb::parallel_for( tbb::blocked_range<int>(0,n,1), [&](tbb::blocked_range<int> r)
    {
        int i;
        for( i = r.begin(); i < r.end(); i++ )
        {
            func(i, a);
        }
    } );
  #elif defined(__DO_SEQ_LAYER__)
    int i;
    for( i = 0; i < n; i++ )
    {
        func(i, a);
    }
  #endif
}

DAAL_EXPORT void _daal_threader_for_blocked(int n, int threads_request, const void* a, daal::functype2 func)
{
  #if defined(__DO_TBB_LAYER__)
    tbb::parallel_for( tbb::blocked_range<int>(0,n,1), [&](tbb::blocked_range<int> r)
    {
        func(r.begin(), r.end()-r.begin(), a);
    } );
  #elif defined(__DO_SEQ_LAYER__)
    func(0, n, a);
  #endif
}

DAAL_EXPORT int _daal_threader_get_max_threads()
{
  #if defined(__DO_TBB_LAYER__)
    return tbb::task_scheduler_init::default_num_threads();
  #elif defined(__DO_SEQ_LAYER__)
    return 1;
  #endif
}

DAAL_EXPORT void* _daal_get_tls_ptr(void* a, daal::tls_functype func)
{
  #if defined(__DO_TBB_LAYER__)
    tbb::enumerable_thread_specific<void*> *p =
        new tbb::enumerable_thread_specific<void*>( [=]()-> void* { return func(a); } );
    return (void*)p;
  #elif defined(__DO_SEQ_LAYER__)
    return func(a);
  #endif
}

DAAL_EXPORT void _daal_del_tls_ptr(void* tlsPtr)
{
  #if defined(__DO_TBB_LAYER__)
    tbb::enumerable_thread_specific<void*> *p =
        static_cast<tbb::enumerable_thread_specific<void*>*>(tlsPtr);
    delete p;
  #elif defined(__DO_SEQ_LAYER__)
  #endif
}

DAAL_EXPORT void* _daal_get_tls_local(void* tlsPtr)
{
  #if defined(__DO_TBB_LAYER__)
    tbb::enumerable_thread_specific<void*> *p =
        static_cast<tbb::enumerable_thread_specific<void*> *>(tlsPtr);
    return p->local();
  #elif defined(__DO_SEQ_LAYER__)
    return tlsPtr;
  #endif
}

DAAL_EXPORT void _daal_reduce_tls(void* tlsPtr, void* a, daal::tls_reduce_functype func)
{
  #if defined(__DO_TBB_LAYER__)
    tbb::enumerable_thread_specific<void*> *p =
        static_cast<tbb::enumerable_thread_specific<void*>*>(tlsPtr);

    for( auto it = p->begin() ; it != p->end() ; ++it )
    {
        func( (*it), a );
    }
  #elif defined(__DO_SEQ_LAYER__)
    func( tlsPtr, a );
  #endif
}

DAAL_EXPORT bool _daal_is_in_parallel()
{
  #if defined(__DO_TBB_LAYER__)
    return tbb::task::self().state() == tbb::task::executing;
  #else
    return false;
  #endif
}
