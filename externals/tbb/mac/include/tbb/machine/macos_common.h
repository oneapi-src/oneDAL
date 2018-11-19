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

#if !defined(__TBB_machine_H) || defined(__TBB_machine_macos_common_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_macos_common_H

#include <sched.h>
#define __TBB_Yield()  sched_yield()

// __TBB_HardwareConcurrency

#include <sys/types.h>
#include <sys/sysctl.h>

static inline int __TBB_macos_available_cpu() {
    int name[2] = {CTL_HW, HW_AVAILCPU};
    int ncpu;
    size_t size = sizeof(ncpu);
    sysctl( name, 2, &ncpu, &size, NULL, 0 );
    return ncpu;
}

#define __TBB_HardwareConcurrency() __TBB_macos_available_cpu()

#ifndef __TBB_full_memory_fence
    // TBB has not recognized the architecture (none of the architecture abstraction
    // headers was included).
    #define __TBB_UnknownArchitecture 1
#endif

#if __TBB_UnknownArchitecture
// Implementation of atomic operations based on OS provided primitives
#include <libkern/OSAtomic.h>

static inline int64_t __TBB_machine_cmpswp8_OsX(volatile void *ptr, int64_t value, int64_t comparand)
{
    __TBB_ASSERT( tbb::internal::is_aligned(ptr,8), "address not properly aligned for macOS* atomics");
    int64_t* address = (int64_t*)ptr;
    while( !OSAtomicCompareAndSwap64Barrier(comparand, value, address) ){
#if __TBB_WORDSIZE==8
        int64_t snapshot = *address;
#else
        int64_t snapshot = OSAtomicAdd64( 0, address );
#endif
        if( snapshot!=comparand ) return snapshot;
    }
    return comparand;
}

#define __TBB_machine_cmpswp8 __TBB_machine_cmpswp8_OsX

#endif /* __TBB_UnknownArchitecture */

#if __TBB_UnknownArchitecture

#ifndef __TBB_WORDSIZE
#define __TBB_WORDSIZE __SIZEOF_POINTER__
#endif

#ifdef __TBB_ENDIANNESS
    // Already determined based on hardware architecture.
#elif __BIG_ENDIAN__
    #define __TBB_ENDIANNESS __TBB_ENDIAN_BIG
#elif __LITTLE_ENDIAN__
    #define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE
#else
    #define __TBB_ENDIANNESS __TBB_ENDIAN_UNSUPPORTED
#endif

/** As this generic implementation has absolutely no information about underlying
    hardware, its performance most likely will be sub-optimal because of full memory
    fence usages where a more lightweight synchronization means (or none at all)
    could suffice. Thus if you use this header to enable TBB on a new platform,
    consider forking it and relaxing below helpers as appropriate. **/
#define __TBB_control_consistency_helper() OSMemoryBarrier()
#define __TBB_acquire_consistency_helper() OSMemoryBarrier()
#define __TBB_release_consistency_helper() OSMemoryBarrier()
#define __TBB_full_memory_fence()          OSMemoryBarrier()

static inline int32_t __TBB_machine_cmpswp4(volatile void *ptr, int32_t value, int32_t comparand)
{
    __TBB_ASSERT( tbb::internal::is_aligned(ptr,4), "address not properly aligned for macOS atomics");
    int32_t* address = (int32_t*)ptr;
    while( !OSAtomicCompareAndSwap32Barrier(comparand, value, address) ){
        int32_t snapshot = *address;
        if( snapshot!=comparand ) return snapshot;
    }
    return comparand;
}

static inline int32_t __TBB_machine_fetchadd4(volatile void *ptr, int32_t addend)
{
    __TBB_ASSERT( tbb::internal::is_aligned(ptr,4), "address not properly aligned for macOS atomics");
    return OSAtomicAdd32Barrier(addend, (int32_t*)ptr) - addend;
}

static inline int64_t __TBB_machine_fetchadd8(volatile void *ptr, int64_t addend)
{
    __TBB_ASSERT( tbb::internal::is_aligned(ptr,8), "address not properly aligned for macOS atomics");
    return OSAtomicAdd64Barrier(addend, (int64_t*)ptr) - addend;
}

#define __TBB_USE_GENERIC_PART_WORD_CAS                     1
#define __TBB_USE_GENERIC_PART_WORD_FETCH_ADD               1
#define __TBB_USE_GENERIC_FETCH_STORE                       1
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE            1
#define __TBB_USE_GENERIC_RELAXED_LOAD_STORE                1
#if __TBB_WORDSIZE == 4
    #define __TBB_USE_GENERIC_DWORD_LOAD_STORE              1
#endif
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1

#endif /* __TBB_UnknownArchitecture */
