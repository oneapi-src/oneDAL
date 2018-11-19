/* file: service_thread_pinner.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

/*
//++
//  Implementation of thread pinner class
//--
*/

#ifndef __SERVICE_THREAD_PINNER_H__
#define __SERVICE_THREAD_PINNER_H__

#include "daal_defines.h"
#if !defined (DAAL_THREAD_PINNING_DISABLED)

#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __FreeBSD__
    #ifndef _GNU_SOURCE
        #define _GNU_SOURCE
    #endif
    #include <sched.h>

    #include <sys/param.h>
    #include <sys/cpuset.h>

    typedef cpuset_t cpu_set_t;

    typedef long int __cpu_mask;
    # define __NCPUBITS (8 * sizeof (__cpu_mask))
    # define __CPUELT(cpu)  ((cpu) / __NCPUBITS)
    # define __CPUMASK(cpu) ((__cpu_mask) 1 << ((cpu) % __NCPUBITS))
    # define __CPU_ALLOC_SIZE(count) \
      ((((count) + __NCPUBITS - 1) / __NCPUBITS) * sizeof (__cpu_mask))
    # define __CPU_ALLOC(count) __sched_cpualloc (count)
    # define __CPU_FREE(cpuset) __sched_cpufree (cpuset)
    # define __CPU_ZERO_S(setsize, cpusetp)            \
      do {                                              \
        size_t __i;                                     \
        size_t __imax = (setsize) / sizeof (__cpu_mask);\
        __cpu_mask *__bits = (cpusetp)->__bits;         \
        for (__i = 0; __i < __imax; ++__i)              \
          __bits[__i] = 0;                              \
      } while (0)
      # define __CPU_SET_S(cpu, setsize, cpusetp)                       \
           (__extension__                                               \
           ({ size_t __cpu = (cpu);                                     \
              __cpu / 8 < (setsize)                                     \
              ? (((__cpu_mask *) ((cpusetp)->__bits))[__CPUELT (__cpu)] \
             |= __CPUMASK (__cpu))                                      \
              : 0; }))
    # define CPU_ALLOC(count) __CPU_ALLOC (count)
    # define CPU_FREE(cpuset) __CPU_FREE (cpuset)
    # define CPU_ALLOC_SIZE(count) __CPU_ALLOC_SIZE (count)
    # define CPU_ZERO_S(setsize, cpusetp) __CPU_ZERO_S (setsize, cpusetp)
    # define CPU_SET_S(cpu, setsize, cpusetp)   __CPU_SET_S (cpu, setsize, cpusetp)
#endif

namespace daal
{
namespace services
{
namespace internal
{

class thread_pinner_task_t
{
public:
    virtual void operator()() = 0;
};

}
}
}

extern "C" {
    DAAL_EXPORT void _thread_pinner_thread_pinner_init(void(int&, int&, int&, int**), void (*deleter)(void*));
    DAAL_EXPORT void _thread_pinner_read_topology();
    DAAL_EXPORT void _thread_pinner_on_scheduler_entry(bool);
    DAAL_EXPORT void _thread_pinner_on_scheduler_exit(bool);
    DAAL_EXPORT void _thread_pinner_execute(daal::services::internal::thread_pinner_task_t& task);

    DAAL_EXPORT int  _thread_pinner_get_status();
    DAAL_EXPORT bool _thread_pinner_get_pinning();
    DAAL_EXPORT bool _thread_pinner_set_pinning(bool p);

    DAAL_EXPORT void* _getThreadPinner(bool create_pinner, void(int&, int&, int&, int**), void (*deleter)(void*));
}

namespace daal
{
namespace services
{
namespace internal
{

class thread_pinner_t
{
public:

    thread_pinner_t(void (*f)(int&, int&, int&, int**), void (*deleter)(void*))
    {
        _thread_pinner_thread_pinner_init(f, deleter);
    }
    void read_topology()
    {
        _thread_pinner_read_topology();
    }
    void on_scheduler_entry( bool p)
    {
        _thread_pinner_on_scheduler_entry(p);
    }
    void on_scheduler_exit( bool p)
    {
        _thread_pinner_on_scheduler_exit(p);
    }
    void execute(thread_pinner_task_t& task)
    {
        _thread_pinner_execute(task);
    }
    int  get_status()
    {
        return _thread_pinner_get_status();
    }
    bool get_pinning()
    {
        return _thread_pinner_get_pinning();
    }
    bool set_pinning(bool p)
    {
        return _thread_pinner_set_pinning(p);
    }
};

inline thread_pinner_t* getThreadPinner(bool create_pinner, void (*read_topo)(int&, int&, int&, int**), void (*deleter)(void*))
{
    return (thread_pinner_t*) _getThreadPinner(create_pinner, read_topo, deleter);
}

}
}
}

#endif /* #if !defined (DAAL_THREAD_PINNING_DISABLED) */
#endif /* #ifndef __SERVICE_THREAD_PINNER_H__ */
