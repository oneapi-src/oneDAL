/* file: threading.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#include "src/threading/threading.h"
#include "services/daal_memory.h"

#if defined(__DO_TBB_LAYER__)
    #define TBB_PREVIEW_GLOBAL_CONTROL 1
    #define TBB_PREVIEW_TASK_ARENA     1

    #include <stdlib.h> // malloc and free
    #include <tbb/tbb.h>
    #include <tbb/spin_mutex.h>
    #include <tbb/scalable_allocator.h>
    #include <tbb/global_control.h>
    #include <tbb/task_arena.h>
    #include "services/daal_atomic_int.h"

    #if defined(TBB_INTERFACE_VERSION) && TBB_INTERFACE_VERSION >= 12002
        #include <tbb/task.h>
    #endif

using namespace daal::services;
#else
    #include "src/externals/service_service.h"
    #include "src/algorithms/service_qsort.h"
#endif

DAAL_EXPORT void * _threaded_scalable_malloc(const size_t size, const size_t alignment)
{
#if defined(__DO_TBB_LAYER__)
    return scalable_aligned_malloc(size, alignment);
#else
    return daal::internal::Service<>::serv_malloc(size, alignment);
#endif
}

DAAL_EXPORT void _threaded_scalable_free(void * ptr)
{
#if defined(__DO_TBB_LAYER__)
    scalable_aligned_free(ptr);
#else
    daal::internal::Service<>::serv_free(ptr);
#endif
}

DAAL_EXPORT void _daal_tbb_task_scheduler_free(void *& globalControl)
{
#if defined(__DO_TBB_LAYER__)
    if (globalControl)
    {
        delete reinterpret_cast<tbb::global_control *>(globalControl);
        globalControl = nullptr;
    }
#endif
}

DAAL_EXPORT size_t _setNumberOfThreads(const size_t numThreads, void ** globalControl)
{
#if defined(__DO_TBB_LAYER__)
    static tbb::spin_mutex mt;
    tbb::spin_mutex::scoped_lock lock(mt);
    if (numThreads != 0)
    {
        _daal_tbb_task_scheduler_free(*globalControl);
        *globalControl = reinterpret_cast<void *>(new tbb::global_control(tbb::global_control::max_allowed_parallelism, numThreads));
        daal::threader_env()->setNumberOfThreads(numThreads);
        return numThreads;
    }
#endif
    daal::threader_env()->setNumberOfThreads(1);
    return 1;
}

DAAL_EXPORT void _daal_threader_for(int n, int threads_request, const void * a, daal::functype func)
{
#if defined(__DO_TBB_LAYER__)
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 1), [&](tbb::blocked_range<int> r) {
        int i;
        for (i = r.begin(); i < r.end(); i++)
        {
            func(i, a);
        }
    });
#elif defined(__DO_SEQ_LAYER__)
    int i;
    for (i = 0; i < n; i++)
    {
        func(i, a);
    }
#endif
}

DAAL_EXPORT void _daal_threader_for_int64(int64_t n, const void * a, daal::functype_int64 func)
{
#if defined(__DO_TBB_LAYER__)
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, n, 1), [&](tbb::blocked_range<int64_t> r) {
        int64_t i;
        for (i = r.begin(); i < r.end(); i++)
        {
            func(i, a);
        }
    });
#elif defined(__DO_SEQ_LAYER__)
    int64_t i;
    for (i = 0; i < n; i++)
    {
        func(i, a);
    }
#endif
}

DAAL_EXPORT void _daal_threader_for_simple(int n, int threads_request, const void * a, daal::functype func)
{
#if defined(__DO_TBB_LAYER__)
    tbb::parallel_for(
        tbb::blocked_range<int>(0, n, 1),
        [&](tbb::blocked_range<int> r) {
            int i;
            for (i = r.begin(); i < r.end(); i++)
            {
                func(i, a);
            }
        },
        tbb::simple_partitioner {});
#elif defined(__DO_SEQ_LAYER__)
    int i;
    for (i = 0; i < n; i++)
    {
        func(i, a);
    }
#endif
}

DAAL_EXPORT void _daal_threader_for_int32ptr(const int * begin, const int * end, const void * a, daal::functype_int32ptr func)
{
#if defined(__DO_TBB_LAYER__)
    tbb::parallel_for(tbb::blocked_range<const int *>(begin, end, 1), [&](tbb::blocked_range<const int *> r) {
        const int * i;
        for (i = r.begin(); i != r.end(); i++)
        {
            func(i, a);
        }
    });
#elif defined(__DO_SEQ_LAYER__)
    const int * i;
    for (i = begin; i != end; ++i)
    {
        func(i, a);
    }
#endif
}

DAAL_EXPORT int64_t _daal_parallel_reduce_int32_int64(int32_t n, int64_t init, const void * a, daal::loop_functype_int32_int64 loop_func,
                                                      const void * b, daal::reduction_functype_int64 reduction_func)
{
#if defined(__DO_TBB_LAYER__)
    return tbb::parallel_reduce(
        tbb::blocked_range<int32_t>(0, n), init,
        [&](const tbb::blocked_range<int32_t> & r, int64_t value_for_reduce) { return loop_func(r.begin(), r.end(), value_for_reduce, a); },
        [&](int64_t x, int64_t y) { return reduction_func(x, y, b); }, tbb::auto_partitioner {});

#elif defined(__DO_SEQ_LAYER__)
    int64_t value_for_reduce = init;
    return loop_func(0, n, value_for_reduce, a);
#endif
}

DAAL_EXPORT int64_t _daal_parallel_reduce_int32_int64_simple(int32_t n, int64_t init, const void * a, daal::loop_functype_int32_int64 loop_func,
                                                             const void * b, daal::reduction_functype_int64 reduction_func)
{
#if defined(__DO_TBB_LAYER__)
    return tbb::parallel_reduce(
        tbb::blocked_range<int32_t>(0, n), init,
        [&](const tbb::blocked_range<int32_t> & r, int64_t value_for_reduce) { return loop_func(r.begin(), r.end(), value_for_reduce, a); },
        [&](int64_t x, int64_t y) { return reduction_func(x, y, b); }, tbb::simple_partitioner {});

#elif defined(__DO_SEQ_LAYER__)
    int64_t value_for_reduce = init;
    return loop_func(0, n, value_for_reduce, a);
#endif
}

DAAL_EXPORT int64_t _daal_parallel_reduce_int32ptr_int64_simple(const int32_t * begin, const int32_t * end, int64_t init, const void * a,
                                                                daal::loop_functype_int32ptr_int64 loop_func, const void * b,
                                                                daal::reduction_functype_int64 reduction_func)
{
#if defined(__DO_TBB_LAYER__)
    return tbb::parallel_reduce(
        tbb::blocked_range<const int32_t *>(begin, end), init,
        [&](const tbb::blocked_range<const int32_t *> & r, int64_t value_for_reduce) { return loop_func(r.begin(), r.end(), value_for_reduce, a); },
        [&](int64_t x, int64_t y) { return reduction_func(x, y, b); }, tbb::simple_partitioner {});

#elif defined(__DO_SEQ_LAYER__)
    int64_t value_for_reduce = init;
    return loop_func(begin, end, value_for_reduce, a);
#endif
}

DAAL_EXPORT void _daal_static_threader_for(size_t n, const void * a, daal::functype_static func)
{
#if defined(__DO_TBB_LAYER__)
    const size_t nthreads           = _daal_threader_get_max_threads();
    const size_t nblocks_per_thread = n / nthreads + !!(n % nthreads);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, nthreads, 1),
        [&](tbb::blocked_range<size_t> r) {
            const size_t tid   = r.begin();
            const size_t begin = tid * nblocks_per_thread;
            const size_t end   = n < begin + nblocks_per_thread ? n : begin + nblocks_per_thread;

            for (size_t i = begin; i < end; ++i)
            {
                func(i, tid, a);
            }
        },
        tbb::static_partitioner());
#elif defined(__DO_SEQ_LAYER__)
    for (size_t i = 0; i < n; i++)
    {
        func(i, 0, a);
    }
#endif
}

template <typename F>
DAAL_EXPORT void _daal_parallel_sort_template(F * begin_p, F * end_p)
{
#if defined(__DO_TBB_LAYER__)
    tbb::parallel_sort(begin_p, end_p);
#elif defined(__DO_SEQ_LAYER__)
    daal::algorithms::internal::qSort<F>(end_p - begin_p, begin_p);
#endif
}

#define DAAL_PARALLEL_SORT_IMPL(TYPE, NAMESUFFIX) \
    DAAL_EXPORT void _daal_parallel_sort_##NAMESUFFIX(TYPE * begin_p, TYPE * end_p) { _daal_parallel_sort_template<TYPE>(begin_p, end_p); }

DAAL_PARALLEL_SORT_IMPL(int, int32)
DAAL_PARALLEL_SORT_IMPL(size_t, uint64)
DAAL_PARALLEL_SORT_IMPL(daal::IdxValType<int>, pair_int32_uint64)
DAAL_PARALLEL_SORT_IMPL(daal::IdxValType<float>, pair_fp32_uint64)
DAAL_PARALLEL_SORT_IMPL(daal::IdxValType<double>, pair_fp64_uint64)

#undef DAAL_PARALLEL_SORT_IMPL

DAAL_EXPORT void _daal_threader_for_blocked(int n, int threads_request, const void * a, daal::functype2 func)
{
#if defined(__DO_TBB_LAYER__)
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 1), [&](tbb::blocked_range<int> r) { func(r.begin(), r.end() - r.begin(), a); });
#elif defined(__DO_SEQ_LAYER__)
    func(0, n, a);
#endif
}

DAAL_EXPORT void _daal_threader_for_optional(int n, int threads_request, const void * a, daal::functype func)
{
#if defined(__DO_TBB_LAYER__)
    if (_daal_is_in_parallel())
    {
        int i;
        for (i = 0; i < n; i++)
        {
            func(i, a);
        }
    }
    else
    {
        _daal_threader_for(n, threads_request, a, func);
    }
#elif defined(__DO_SEQ_LAYER__)
    _daal_threader_for(n, threads_request, a, func);
#endif
}

DAAL_EXPORT void _daal_threader_for_break(int n, int threads_request, const void * a, daal::functype_break func)
{
#if defined(__DO_TBB_LAYER__)
    tbb::task_group_context context;
    tbb::parallel_for(
        tbb::blocked_range<int>(0, n, 1),
        [&](tbb::blocked_range<int> r) {
            int i;
            for (i = r.begin(); i < r.end(); ++i)
            {
                bool needBreak = false;
                func(i, needBreak, a);
                if (needBreak) context.cancel_group_execution();
            }
        },
        context);
#elif defined(__DO_SEQ_LAYER__)
    int i;
    for (i = 0; i < n; ++i)
    {
        bool needBreak = false;
        func(i, needBreak, a);
        if (needBreak) break;
    }
#endif
}

DAAL_EXPORT int _daal_threader_get_max_threads()
{
#if defined(__DO_TBB_LAYER__)
    return tbb::this_task_arena::max_concurrency();
#elif defined(__DO_SEQ_LAYER__)
    return 1;
#endif
}

DAAL_EXPORT int _daal_threader_get_current_thread_index()
{
#if defined(__DO_TBB_LAYER__)
    return tbb::this_task_arena::current_thread_index();
#elif defined(__DO_SEQ_LAYER__)
    return 0;
#endif
}

DAAL_EXPORT void * _daal_get_tls_ptr(void * a, daal::tls_functype func)
{
#if defined(__DO_TBB_LAYER__)
    tbb::enumerable_thread_specific<void *> * p = new tbb::enumerable_thread_specific<void *>([=]() -> void * { return func(a); });
    return (void *)p;
#elif defined(__DO_SEQ_LAYER__)
    return func(a);
#endif
}

DAAL_EXPORT void _daal_del_tls_ptr(void * tlsPtr)
{
#if defined(__DO_TBB_LAYER__)
    tbb::enumerable_thread_specific<void *> * p = static_cast<tbb::enumerable_thread_specific<void *> *>(tlsPtr);
    delete p;
#elif defined(__DO_SEQ_LAYER__)
#endif
}

DAAL_EXPORT void * _daal_get_tls_local(void * tlsPtr)
{
#if defined(__DO_TBB_LAYER__)
    tbb::enumerable_thread_specific<void *> * p = static_cast<tbb::enumerable_thread_specific<void *> *>(tlsPtr);
    return p->local();
#elif defined(__DO_SEQ_LAYER__)
    return tlsPtr;
#endif
}

DAAL_EXPORT void _daal_reduce_tls(void * tlsPtr, void * a, daal::tls_reduce_functype func)
{
#if defined(__DO_TBB_LAYER__)
    tbb::enumerable_thread_specific<void *> * p = static_cast<tbb::enumerable_thread_specific<void *> *>(tlsPtr);

    for (auto it = p->begin(); it != p->end(); ++it)
    {
        func((*it), a);
    }
#elif defined(__DO_SEQ_LAYER__)
    func(tlsPtr, a);
#endif
}

DAAL_EXPORT void _daal_parallel_reduce_tls(void * tlsPtr, void * a, daal::tls_reduce_functype func)
{
#if defined(__DO_TBB_LAYER__)
    size_t n                                    = 0;
    tbb::enumerable_thread_specific<void *> * p = static_cast<tbb::enumerable_thread_specific<void *> *>(tlsPtr);

    for (auto it = p->begin(); it != p->end(); ++it, ++n)
        ;
    if (n)
    {
        typedef void * mptr;
        mptr * aDataPtr = (mptr *)(::malloc(sizeof(mptr) * n));
        if (aDataPtr)
        {
            size_t i = 0;
            for (auto it = p->begin(); it != p->end(); ++it) aDataPtr[i++] = *it;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n, 1), [&](tbb::blocked_range<size_t> r) {
                for (size_t i = r.begin(); i < r.end(); i++) func(aDataPtr[i], a);
            });
            ::free(aDataPtr);
        }
    }
#elif defined(__DO_SEQ_LAYER__)
    func(tlsPtr, a);
#endif
}

DAAL_EXPORT void * _daal_new_mutex()
{
#if defined(__DO_TBB_LAYER__)
    return new tbb::spin_mutex();
#elif defined(__DO_SEQ_LAYER__)
    return NULL;
#endif
}

DAAL_EXPORT void _daal_lock_mutex(void * mutexPtr)
{
#if defined(__DO_TBB_LAYER__)
    static_cast<tbb::spin_mutex *>(mutexPtr)->lock();
#endif
}

DAAL_EXPORT void _daal_unlock_mutex(void * mutexPtr)
{
#if defined(__DO_TBB_LAYER__)
    static_cast<tbb::spin_mutex *>(mutexPtr)->unlock();
#endif
}

DAAL_EXPORT void _daal_del_mutex(void * mutexPtr)
{
#if defined(__DO_TBB_LAYER__)
    delete static_cast<tbb::spin_mutex *>(mutexPtr);
#endif
}

DAAL_EXPORT bool _daal_is_in_parallel()
{
#if defined(__DO_TBB_LAYER__)
    #if defined(TBB_INTERFACE_VERSION) && TBB_INTERFACE_VERSION >= 12002
    return tbb::task::current_context() != nullptr;
    #else
    return tbb::task::self().state() == tbb::task::executing;
    #endif
#else
    return false;
#endif
}

DAAL_EXPORT void * _daal_threader_env()
{
    static daal::ThreaderEnvironment env;
    return &env;
}

#if defined(__DO_TBB_LAYER__)
template <typename T, typename Key, typename Pred>
//Returns an index of the first element in the range[ar, ar + n) that is not less than(i.e.greater or equal to) value.
size_t lower_bound(size_t n, const T * ar, const Key & value)
{
    const T * first = ar;
    while (n > 0)
    {
        auto it   = first;
        auto step = (n >> 1);
        it += step;
        if (Pred::less(*it, value))
        {
            first = ++it;
            n -= step + 1;
        }
        else
            n = step;
    }
    return first - ar;
}

class SimpleAllocator
{
public:
    static void * alloc(size_t n) { return ::malloc(n); }
    static void free(void * p) { ::free(p); }
};

template <class T, class Allocator>
class Collection
{
public:
    /**
    *  Default constructor. Sets the size and capacity to 0.
    */
    Collection() : _array(NULL), _size(0), _capacity(0) {}

    /**
    *  Destructor
    */
    virtual ~Collection()
    {
        for (size_t i = 0; i < _capacity; i++) _array[i].~T();
        Allocator::free(_array);
    }

    /**
    *  Element access
    *  \param[in] index Index of an accessed element
    *  \return    Reference to the element
    */
    T & operator[](size_t index) { return _array[index]; }

    /**
    *  Const element access
    *  \param[in] index Index of an accessed element
    *  \return    Reference to the element
    */
    const T & operator[](size_t index) const { return _array[index]; }

    /**
    *  Size of a collection
    *  \return Size of the collection
    */
    size_t size() const { return _size; }

    /**
    *  Changes the size of a storage
    *  \param[in] newCapacity Size of a new storage.
    */
    bool resize(size_t newCapacity)
    {
        if (newCapacity <= _capacity)
        {
            return true;
        }
        T * newArray = (T *)Allocator::alloc(sizeof(T) * newCapacity);
        if (!newArray)
        {
            return false;
        }
        for (size_t i = 0; i < newCapacity; i++)
        {
            T * elementMemory = &(newArray[i]);
            ::new (elementMemory) T;
        }

        size_t minSize = newCapacity < _size ? newCapacity : _size;
        for (size_t i = 0; i < minSize; i++) newArray[i] = _array[i];

        for (size_t i = 0; i < _capacity; i++) _array[i].~T();

        Allocator::free(_array);
        _array    = newArray;
        _capacity = newCapacity;
        return true;
    }

    /**
    *  Clears a collection: removes an array, sets the size and capacity to 0
    */
    void clear()
    {
        for (size_t i = 0; i < _capacity; i++) _array[i].~T();

        Allocator::free(_array);
        _array    = NULL;
        _size     = 0;
        _capacity = 0;
    }

    /**
    *  Insert an element into a position
    *  \param[in] pos Position to set
    *  \param[in] x   Element to set
    */
    bool insert(const size_t pos, const T & x)
    {
        if (pos > this->size()) return true;

        size_t newSize = 1 + this->size();
        if (newSize > _capacity)
        {
            if (!_resize()) return false;
        }

        size_t tail = _size - pos;
        for (size_t i = 0; i < tail; i++) _array[_size - i] = _array[_size - 1 - i];
        _array[pos] = x;
        _size       = newSize;
        return true;
    }

    /**
    *  Erase an element from a position
    *  \param[in] pos Position to erase
    */
    void erase(size_t pos)
    {
        if (pos >= this->size()) return;
        _size--;
        for (size_t i = 0; i < _size - pos; i++) _array[pos + i] = _array[pos + 1 + i];
    }

private:
    static const size_t _default_capacity = 16;
    bool _resize()
    {
        size_t newCapacity = 2 * _capacity;
        if (_capacity == 0) newCapacity = _default_capacity;
        return resize(newCapacity);
    }

protected:
    T * _array;
    size_t _size;
    size_t _capacity;
};

    #if _WIN32 || _WIN64
typedef DWORD ThreadId;
ThreadId getCurrentThreadId()
{
    return ::GetCurrentThreadId();
}
    #else
typedef pthread_t ThreadId;
ThreadId getCurrentThreadId()
{
    return pthread_self();
}
    #endif // _WIN32||_WIN64

class LocalStorage
{
public:
    LocalStorage(void * a, daal::tls_functype func) : _a(a), _func(func) {}
    LocalStorage(const LocalStorage & o) = delete;
    LocalStorage & operator=(const LocalStorage & o) = delete;

    void * get()
    {
        auto tid = getCurrentThreadId();
        {
            tbb::spin_mutex::scoped_lock lock(_mt);
            size_t i;
            if (findFree(tid, i))
            {
                void * res = _free[i].value;
                addUsed(_free[i]);
                _free.erase(i);
                return res;
            }
        }
        Pair p(tid, _func(_a));
        if (p.value)
        {
            tbb::spin_mutex::scoped_lock lock(_mt);
            addUsed(p);
        }
        return p.value;
    }

    void release(void * data)
    {
        tbb::spin_mutex::scoped_lock lock(_mt);
        size_t i = findUsed(data);
        addFree(_used[i]);
        _used.erase(i);
    }

    void reduce(void * a, daal::tls_reduce_functype func)
    {
        tbb::spin_mutex::scoped_lock lock(_mt);
        for (size_t i = 0; i < _free.size(); ++i) func(_free[i].value, a);
        for (size_t i = 0; i < _used.size(); ++i) func(_used[i].value, a);
        _free.clear();
        _used.clear();
    }

private:
    struct Pair
    {
        Pair() : tid(0), value(NULL) {}
        Pair(const ThreadId & id, void * v) : tid(id), value(v) {}
        Pair(const Pair & o) : tid(o.tid), value(o.value) {}
        Pair & operator=(const Pair & o)
        {
            tid   = o.tid;
            value = o.value;
            return *this;
        }

        ThreadId tid;
        void * value;
    };
    struct CompareByTid
    {
        static bool less(const Pair & p, const ThreadId & tid) { return p.tid < tid; }
    };
    struct CompareByValue
    {
        static bool less(const Pair & p, const void * val) { return p.value < val; }
    };

    bool findFree(const ThreadId & tid, size_t & i) const
    {
        if (!_free.size()) return false;
        i = lower_bound<Pair, ThreadId, CompareByTid>(_free.size(), &_free[0], tid);
        if (i == _free.size()) --i;
        return true;
    }

    size_t findUsed(void * data) const
    {
        size_t i = lower_bound<Pair, void *, CompareByValue>(_used.size(), &_used[0], data);
        //DAAL_ASSERT(i < _used.size());
        return i;
    }

    void addFree(const Pair & p)
    {
        size_t i = lower_bound<Pair, ThreadId, CompareByTid>(_free.size(), &_free[0], p.tid);
        _free.insert(i, p);
    }

    void addUsed(const Pair & p)
    {
        size_t i = lower_bound<Pair, void *, CompareByValue>(_used.size(), &_used[0], p.value);
        _used.insert(i, p);
    }

private:
    void * _a;
    daal::tls_functype _func;
    Collection<Pair, SimpleAllocator> _free; //sorted by tid
    Collection<Pair, SimpleAllocator> _used; //sorted by value
    tbb::spin_mutex _mt;
};

DAAL_EXPORT void * _daal_get_ls_ptr(void * a, daal::tls_functype func)
{
    return new LocalStorage(a, func);
}

DAAL_EXPORT void * _daal_get_ls_local(void * lsPtr)
{
    return ((LocalStorage *)lsPtr)->get();
}

DAAL_EXPORT void _daal_reduce_ls(void * lsPtr, void * a, daal::tls_reduce_functype func)
{
    ((LocalStorage *)lsPtr)->reduce(a, func);
}

DAAL_EXPORT void _daal_del_ls_ptr(void * lsPtr)
{
    delete ((LocalStorage *)lsPtr);
}

DAAL_EXPORT void _daal_release_ls_local(void * lsPtr, void * p)
{
    ((LocalStorage *)lsPtr)->release(p);
}

DAAL_EXPORT void * _daal_new_task_group()
{
    return new tbb::task_group();
}

DAAL_EXPORT void _daal_del_task_group(void * taskGroupPtr)
{
    delete (tbb::task_group *)taskGroupPtr;
}

DAAL_EXPORT void _daal_run_task_group(void * taskGroupPtr, daal::task * t)
{
    struct shared_task
    {
        typedef Atomic<int> RefCounterType;

        shared_task(daal::task & t) : _t(t), _nRefs(nullptr)
        {
            _nRefs = new RefCounterType;
            (*_nRefs).set(1);
        }

        shared_task(const shared_task & o) : _t(o._t), _nRefs(o._nRefs) { (*_nRefs).inc(); }

        ~shared_task()
        {
            if (_nRefs && !(*_nRefs).dec())
            {
                _t.destroy();
                delete _nRefs;
            }
        }

        void operator()() const { _t.run(); }

        daal::task & _t;
        RefCounterType * _nRefs;

    private:
        shared_task & operator=(const shared_task &);
    };
    tbb::task_group * group = (tbb::task_group *)taskGroupPtr;
    group->run(shared_task(*t));
}

DAAL_EXPORT void _daal_wait_task_group(void * taskGroupPtr)
{
    ((tbb::task_group *)taskGroupPtr)->wait();
}

#else
DAAL_EXPORT void * _daal_get_ls_ptr(void * a, daal::tls_functype func)
{
    return func(a);
}

DAAL_EXPORT void * _daal_get_ls_local(void * lsPtr)
{
    return lsPtr;
}

DAAL_EXPORT void _daal_reduce_ls(void * lsPtr, void * a, daal::tls_reduce_functype func)
{
    func(lsPtr, a);
}

DAAL_EXPORT void _daal_del_ls_ptr(void * lsPtr) {}

DAAL_EXPORT void _daal_release_ls_local(void * lsPtr, void * p) {}

DAAL_EXPORT void * _daal_new_task_group()
{
    return nullptr;
}

DAAL_EXPORT void _daal_del_task_group(void * taskGroupPtr) {}

DAAL_EXPORT void _daal_run_task_group(void * taskGroupPtr, daal::task * task) {}

DAAL_EXPORT void _daal_wait_task_group(void * taskGroupPtr) {}

#endif

namespace daal
{}
