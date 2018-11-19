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

#ifndef __TBB_queuing_mutex_H
#define __TBB_queuing_mutex_H

#include <cstring>
#include "atomic.h"
#include "tbb_profiling.h"

namespace tbb {

//! Queuing mutex with local-only spinning.
/** @ingroup synchronization */
class queuing_mutex : internal::mutex_copy_deprecated_and_disabled {
public:
    //! Construct unacquired mutex.
    queuing_mutex() {
        q_tail = NULL;
#if TBB_USE_THREADING_TOOLS
        internal_construct();
#endif
    }

    //! The scoped locking pattern
    /** It helps to avoid the common problem of forgetting to release lock.
        It also nicely provides the "node" for queuing locks. */
    class scoped_lock: internal::no_copy {
        //! Initialize fields to mean "no lock held".
        void initialize() {
            mutex = NULL;
#if TBB_USE_ASSERT
            internal::poison_pointer(next);
#endif /* TBB_USE_ASSERT */
        }

    public:
        //! Construct lock that has not acquired a mutex.
        /** Equivalent to zero-initialization of *this. */
        scoped_lock() {initialize();}

        //! Acquire lock on given mutex.
        scoped_lock( queuing_mutex& m ) {
            initialize();
            acquire(m);
        }

        //! Release lock (if lock is held).
        ~scoped_lock() {
            if( mutex ) release();
        }

        //! Acquire lock on given mutex.
        void __TBB_EXPORTED_METHOD acquire( queuing_mutex& m );

        //! Acquire lock on given mutex if free (i.e. non-blocking)
        bool __TBB_EXPORTED_METHOD try_acquire( queuing_mutex& m );

        //! Release lock.
        void __TBB_EXPORTED_METHOD release();

    private:
        //! The pointer to the mutex owned, or NULL if not holding a mutex.
        queuing_mutex* mutex;

        //! The pointer to the next competitor for a mutex
        scoped_lock *next;

        //! The local spin-wait variable
        /** Inverted (0 - blocked, 1 - acquired the mutex) for the sake of
            zero-initialization.  Defining it as an entire word instead of
            a byte seems to help performance slightly. */
        uintptr_t going;
    };

    void __TBB_EXPORTED_METHOD internal_construct();

    // Mutex traits
    static const bool is_rw_mutex = false;
    static const bool is_recursive_mutex = false;
    static const bool is_fair_mutex = true;

private:
    //! The last competitor requesting the lock
    atomic<scoped_lock*> q_tail;

};

__TBB_DEFINE_PROFILING_SET_NAME(queuing_mutex)

} // namespace tbb

#endif /* __TBB_queuing_mutex_H */
