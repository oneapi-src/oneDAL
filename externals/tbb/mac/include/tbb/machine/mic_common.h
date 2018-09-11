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

#ifndef __TBB_mic_common_H
#define __TBB_mic_common_H

#ifndef __TBB_machine_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#if ! __TBB_DEFINE_MIC
    #error mic_common.h should be included only when building for Intel(R) Many Integrated Core Architecture
#endif

#ifndef __TBB_PREFETCHING
#define __TBB_PREFETCHING 1
#endif
#if __TBB_PREFETCHING
#include <immintrin.h>
#define __TBB_cl_prefetch(p) _mm_prefetch((const char*)p, _MM_HINT_T1)
#define __TBB_cl_evict(p) _mm_clevict(p, _MM_HINT_T1)
#endif

/** Intel(R) Many Integrated Core Architecture does not support mfence and pause instructions **/
#define __TBB_full_memory_fence() __asm__ __volatile__("lock; addl $0,(%%rsp)":::"memory")
#define __TBB_Pause(x) _mm_delay_32(16*(x))
#define __TBB_STEALING_PAUSE 1500/16
#include <sched.h>
#define __TBB_Yield() sched_yield()

/** Specifics **/
#define __TBB_STEALING_ABORT_ON_CONTENTION 1
#define __TBB_YIELD2P 1
#define __TBB_HOARD_NONLOCAL_TASKS 1

#if ! ( __FreeBSD__ || __linux__ )
    #error Intel(R) Many Integrated Core Compiler does not define __FreeBSD__ or __linux__ anymore. Check for the __TBB_XXX_BROKEN defined under __FreeBSD__ or __linux__.
#endif /* ! ( __FreeBSD__ || __linux__ ) */

#endif /* __TBB_mic_common_H */
