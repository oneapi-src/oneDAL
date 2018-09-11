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

// TODO: revise by comparing with mac_ppc.h

#if !defined(__TBB_machine_H) || defined(__TBB_machine_ibm_aix51_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_ibm_aix51_H

#define __TBB_WORDSIZE 8
#define __TBB_ENDIANNESS __TBB_ENDIAN_BIG // assumption based on operating system

#include <stdint.h>
#include <unistd.h>
#include <sched.h>

extern "C" {
int32_t __TBB_machine_cas_32 (volatile void* ptr, int32_t value, int32_t comparand);
int64_t __TBB_machine_cas_64 (volatile void* ptr, int64_t value, int64_t comparand);
void __TBB_machine_flush ();
void __TBB_machine_lwsync ();
void __TBB_machine_isync ();
}

// Mapping of old entry point names retained for the sake of backward binary compatibility
#define __TBB_machine_cmpswp4 __TBB_machine_cas_32
#define __TBB_machine_cmpswp8 __TBB_machine_cas_64

#define __TBB_Yield() sched_yield()

#define __TBB_USE_GENERIC_PART_WORD_CAS                     1
#define __TBB_USE_GENERIC_FETCH_ADD                         1
#define __TBB_USE_GENERIC_FETCH_STORE                       1
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE            1
#define __TBB_USE_GENERIC_RELAXED_LOAD_STORE                1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1

#if __GNUC__
    #define __TBB_control_consistency_helper() __asm__ __volatile__( "isync": : :"memory")
    #define __TBB_acquire_consistency_helper() __asm__ __volatile__("lwsync": : :"memory")
    #define __TBB_release_consistency_helper() __asm__ __volatile__("lwsync": : :"memory")
    #define __TBB_full_memory_fence()          __asm__ __volatile__(  "sync": : :"memory")
#else
    // IBM C++ Compiler does not support inline assembly
    // TODO: Since XL 9.0 or earlier GCC syntax is supported. Replace with more
    //       lightweight implementation (like in mac_ppc.h)
    #define __TBB_control_consistency_helper() __TBB_machine_isync ()
    #define __TBB_acquire_consistency_helper() __TBB_machine_lwsync ()
    #define __TBB_release_consistency_helper() __TBB_machine_lwsync ()
    #define __TBB_full_memory_fence()          __TBB_machine_flush ()
#endif
