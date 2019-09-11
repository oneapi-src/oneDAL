/* file: service_allocators.h */
/*******************************************************************************
* Copyright 2015-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#ifndef __SERVICE_ALLOCATORS_H__
#define __SERVICE_ALLOCATORS_H__

#include "service_utils.h"
#include "service_memory.h"
#include "service_type_traits.h"

namespace daal
{
namespace services
{
namespace internal
{

template<typename T, CpuType cpu, typename ...Args>
inline void constructAt(T *ptr, Args &&...args)
{
    ::new ( static_cast<void *>(ptr) ) T(forward<cpu, Args>(args) ...);
}

template<typename T, CpuType cpu>
inline void destroyAt(T *ptr)
{
    ptr->~T();
}

template<typename T, CpuType cpu, typename ...Args>
void constructRange(T *first, T *last, Args &&...args)
{
    for (T *ptr = first; ptr != last; ++ptr)
    {
        constructAt<T, cpu, Args...>(ptr, args...);
    }
}

template<typename T, CpuType cpu>
void destroyRange(T *first, T *last)
{
    for (T *ptr = first; ptr != last; ++ptr)
    {
        destroyAt<T, cpu>(ptr);
    }
}


/* CPU specific allocators */

template<typename T, CpuType cpu>
struct DefaultAllocator
{
    static T *allocate(size_t n) { return new T[n]; }
    static void deallocate(T *ptr) { delete[] ptr; }
};

template<typename T, CpuType cpu>
struct DAALMalloc
{
    static T *allocate(size_t n) { return service_malloc<T, cpu>(n); }
    static void deallocate(T *ptr) { service_free<T, cpu>(ptr); }
};

template<typename T, CpuType cpu>
struct DAALCalloc
{
    static T *allocate(size_t n) { return service_calloc<T, cpu>(n); }
    static void deallocate(T *ptr) { service_free<T, cpu>(ptr); }
};

template<typename T, CpuType cpu>
struct ScalableMalloc
{
    static T *allocate(size_t n) { return service_scalable_malloc<T, cpu>(n); }
    static void deallocate(T *ptr) { service_scalable_free<T, cpu>(ptr); }
};

template<typename T, CpuType cpu>
struct ScalableCalloc
{
    static T *allocate(size_t n) { return service_scalable_calloc<T, cpu>(n); }
    static void deallocate(T *ptr) { service_scalable_free<T, cpu>(ptr); }
};


/* CPU specific deleters */

template<typename T, CpuType cpu>
struct DefaultDeleter
{
    void operator () (T *ptr) { delete ptr; }
};

template<typename T, CpuType cpu>
struct EmptyDeleter
{
    void operator () (T *ptr) { }
};


/* Construction policy that does call explicitly constructor/destructor for type T */
template<typename T, CpuType cpu>
struct DoConstruct
{
    static void construct(T *begin, T *end)
    {
        constructRange<T, cpu>(begin, end);
    }

    static void destroy(T *begin, T *end)
    {
        destroyRange<T, cpu>(begin, end);
    }
};

/* Construction policy that doesn't call constructor/destructor */
template<typename T, CpuType cpu>
struct DoNotConstruct
{
    static void construct(T *begin, T *end) { }
    static void destroy(T *begin, T *end) { }
};

/* Allows to detect primitive and non-primitive types automatically
 * and apply an appropriate construction policy  */
template<typename T, CpuType cpu, bool isPrimitive = IsPrimitiveType<T, cpu>::value>
struct DefaultConstructionPolicy { };

/* Enables DoNotConstruct policy for all primitive types */
template<typename T, CpuType cpu>
struct DefaultConstructionPolicy<T, cpu, /* isPrimitive = */ true> : DoNotConstruct<T, cpu> { };

/* Enables DoConstruct policy for all 'complex' types */
template<typename T, CpuType cpu>
struct DefaultConstructionPolicy<T, cpu, /* isPrimitive = */ false> : DoConstruct<T, cpu> { };

} // namespace internal
} // namespace services
} // namespace daal

#endif
