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

#if !defined(__TBB_machine_H) || defined(__TBB_msvc_armv7_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_msvc_armv7_H

#include <intrin.h>
#include <float.h>

#define __TBB_WORDSIZE 4

#define __TBB_ENDIANNESS __TBB_ENDIAN_UNSUPPORTED

#if defined(TBB_WIN32_USE_CL_BUILTINS)
// We can test this on _M_IX86
#pragma intrinsic(_ReadWriteBarrier)
#pragma intrinsic(_mm_mfence)
#define __TBB_compiler_fence()    _ReadWriteBarrier()
#define __TBB_full_memory_fence() _mm_mfence()
#define __TBB_control_consistency_helper() __TBB_compiler_fence()
#define __TBB_acquire_consistency_helper() __TBB_compiler_fence()
#define __TBB_release_consistency_helper() __TBB_compiler_fence()
#else
//Now __dmb(_ARM_BARRIER_SY) is used for both compiler and memory fences
//This might be changed later after testing
#define __TBB_compiler_fence()    __dmb(_ARM_BARRIER_SY)
#define __TBB_full_memory_fence() __dmb(_ARM_BARRIER_SY)
#define __TBB_control_consistency_helper() __TBB_compiler_fence()
#define __TBB_acquire_consistency_helper() __TBB_full_memory_fence()
#define __TBB_release_consistency_helper() __TBB_full_memory_fence()
#endif

//--------------------------------------------------
// Compare and swap
//--------------------------------------------------

/**
 * Atomic CAS for 32 bit values, if *ptr==comparand, then *ptr=value, returns *ptr
 * @param ptr pointer to value in memory to be swapped with value if *ptr==comparand
 * @param value value to assign *ptr to if *ptr==comparand
 * @param comparand value to compare with *ptr
 * @return value originally in memory at ptr, regardless of success
*/

#define __TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(S,T,F)                                               \
inline T __TBB_machine_cmpswp##S( volatile void *ptr, T value, T comparand ) {                   \
    return _InterlockedCompareExchange##F(reinterpret_cast<volatile T *>(ptr),value,comparand);  \
}                                                                                                \

#define __TBB_MACHINE_DEFINE_ATOMICS_FETCHADD(S,T,F)                                             \
inline T __TBB_machine_fetchadd##S( volatile void *ptr, T value ) {                              \
    return _InterlockedExchangeAdd##F(reinterpret_cast<volatile T *>(ptr),value);                \
}                                                                                                \

__TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(1,char,8)
__TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(2,short,16)
__TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(4,long,)
__TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(8,__int64,64)
__TBB_MACHINE_DEFINE_ATOMICS_FETCHADD(4,long,)
#if defined(TBB_WIN32_USE_CL_BUILTINS)
// No _InterlockedExchangeAdd64 intrinsic on _M_IX86
#define __TBB_64BIT_ATOMICS 0
#else
__TBB_MACHINE_DEFINE_ATOMICS_FETCHADD(8,__int64,64)
#endif

inline void __TBB_machine_pause (int32_t delay )
{
    while(delay>0)
    {
        __TBB_compiler_fence();
        delay--;
    }
}

// API to retrieve/update FPU control setting
#define __TBB_CPU_CTL_ENV_PRESENT 1

namespace tbb {
namespace internal {

template <typename T, size_t S>
struct machine_load_store_relaxed {
    static inline T load ( const volatile T& location ) {
        const T value = location;

        /*
        * An extra memory barrier is required for errata #761319
        * Please see http://infocenter.arm.com/help/topic/com.arm.doc.uan0004a
        */
        __TBB_acquire_consistency_helper();
        return value;
    }

    static inline void store ( volatile T& location, T value ) {
        location = value;
    }
};

class cpu_ctl_env {
private:
    unsigned int my_ctl;
public:
    bool operator!=( const cpu_ctl_env& ctl ) const { return my_ctl != ctl.my_ctl; }
    void get_env() { my_ctl = _control87(0, 0); }
    void set_env() const { _control87( my_ctl, ~0U ); }
};

} // namespace internal
} // namespaces tbb

// Machine specific atomic operations
#define __TBB_CompareAndSwap4(P,V,C) __TBB_machine_cmpswp4(P,V,C)
#define __TBB_CompareAndSwap8(P,V,C) __TBB_machine_cmpswp8(P,V,C)
#define __TBB_Pause(V) __TBB_machine_pause(V)

// Use generics for some things
#define __TBB_USE_FETCHSTORE_AS_FULL_FENCED_STORE               1
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE                1
#define __TBB_USE_GENERIC_PART_WORD_FETCH_ADD                   1
#define __TBB_USE_GENERIC_PART_WORD_FETCH_STORE                 1
#define __TBB_USE_GENERIC_FETCH_STORE                           1
#define __TBB_USE_GENERIC_DWORD_LOAD_STORE                      1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE     1

#if defined(TBB_WIN32_USE_CL_BUILTINS)
#if !__TBB_WIN8UI_SUPPORT
extern "C" __declspec(dllimport) int __stdcall SwitchToThread( void );
#define __TBB_Yield()  SwitchToThread()
#else
#include<thread>
#define __TBB_Yield()  std::this_thread::yield()
#endif
#else
#define __TBB_Yield() __yield()
#endif

// Machine specific atomic operations
#define __TBB_AtomicOR(P,V)     __TBB_machine_OR(P,V)
#define __TBB_AtomicAND(P,V)    __TBB_machine_AND(P,V)

template <typename T1,typename T2>
inline void __TBB_machine_OR( T1 *operand, T2 addend ) {
    _InterlockedOr((long volatile *)operand, (long)addend);
}

template <typename T1,typename T2>
inline void __TBB_machine_AND( T1 *operand, T2 addend ) {
    _InterlockedAnd((long volatile *)operand, (long)addend);
}

