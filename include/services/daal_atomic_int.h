/* file: daal_atomic_int.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Declaration of class for atomic operations with int
//--
*/

#ifndef __DAAL_ATOMIC_INT_H__
#define __DAAL_ATOMIC_INT_H__

#if defined(_WIN32) || defined(_WIN64)
    #include <intrin.h>
#endif

/*
#if !defined(TBB_SUPPRESS_DEPRECATED_MESSAGES)
  #define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#endif
#if !defined(__TBB_LEGACY_MODE)
  #define __TBB_LEGACY_MODE 1
#endif

#include "tbb/tbb.h"
#include "tbb/atomic.h"
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif
*/
#include "services/daal_defines.h"

namespace daal
{
namespace services
{
namespace interface1
{
/**
 * @ingroup memory
 * @{
 */
/**
 * <a name="DAAL-CLASS-SERVICES__ATOMIC"></a>
 * \brief Class that represents an atomic object
 *
 * \tparam dataType Data type of the atomic object
 */
template <typename dataType = int>
class DAAL_EXPORT Atomic
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    inline dataType inc();

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    inline dataType dec();

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    inline void set(dataType value);

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    inline dataType get() const;

    /**
     * Constructs an atomic object
     */
    Atomic() = default;

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(dataType value) : my_storage(value) {}

    /** Destructor */
    ~Atomic() = default;

protected:
    dataType my_storage;

private:
    Atomic(const Atomic &);
    Atomic & operator=(const Atomic &);
};

#if defined(_WIN32)

template <typename dataType>
inline dataType Atomic<dataType>::inc()
{
    return _InterlockedExchangeAdd((long *)&my_storage, 1);
}
template <typename dataType>
inline dataType Atomic<dataType>::dec()
{
    return _InterlockedExchangeAdd((long *)&my_storage, -1);
}

#if defined(_WIN64)
template<>
inline size_t Atomic<size_t>::inc()
{
    return _InterlockedExchangeAdd64((__int64 *)&my_storage, 1);
}
template<>
inline size_t Atomic<size_t>::dec()
{
    return _InterlockedExchangeAdd64((__int64 *)&my_storage, -1);
}
#endif

template <typename dataType>
inline void Atomic<dataType>::set(dataType value)
{
    _ReadWriteBarrier();
    my_storage = value;
}

template <typename dataType>
inline dataType Atomic<dataType>::get() const
{
    dataType to_return = my_storage;
    _ReadWriteBarrier();
    return to_return;
}
#endif

/** @} */

} // namespace interface1

using interface1::Atomic;

typedef Atomic<int> AtomicInt;

} // namespace services
} // namespace daal

#endif
