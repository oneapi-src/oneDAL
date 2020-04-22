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

#if defined(_WIN32)
template <typename dataType>
class DAAL_EXPORT Atomic
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    inline dataType inc()
    {
        DAAL_ASSERT(sizeof(my_storage)==sizeof(long))
        DAAL_ASSERT(sizeof(my_storage)==sizeof(size_t))
        return (dataType)(_InterlockedExchangeAdd((long *)(&my_storage), 1) + 1);
    }

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    inline dataType dec()
    {
        DAAL_ASSERT(sizeof(my_storage)==sizeof(long))
        DAAL_ASSERT(sizeof(my_storage)==sizeof(size_t))
        return (dataType)(_InterlockedExchangeAdd((long *)(&my_storage), -1) - 1);
    }

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    inline void set(dataType value)
    {
        _ReadWriteBarrier();
        my_storage = value;
    }

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    inline dataType get() const
    {
        dataType to_return = my_storage;
        _ReadWriteBarrier();
        return to_return;
    }

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
#endif // _WIN32

#if defined(_WIN64)
/**
 * <a name="DAAL-CLASS-SERVICES__ATOMIC"></a>
 * \brief Class that represents an atomic object
 *
 * \tparam dataType Data type of the atomic object
 */
template <typename dataType>
class DAAL_EXPORT Atomic
{
};

template <>
class DAAL_EXPORT Atomic<int>
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    inline int inc()
    {
        DAAL_ASSERT(sizeof(my_storage)==sizeof(long))
        return (int)(_InterlockedExchangeAdd((long *)(&my_storage), 1) + 1);
    }

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    inline int dec()
    {
        DAAL_ASSERT(sizeof(my_storage)==sizeof(long))
        return (int)(_InterlockedExchangeAdd((long *)(&my_storage), -1) - 1);
    }

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    inline void set(int value)
    {
        _ReadWriteBarrier();
        my_storage = value;
    }

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    inline int get() const
    {
        int to_return = my_storage;
        _ReadWriteBarrier();
        return to_return;
    }

    /**
     * Constructs an atomic object
     */
    Atomic() = default;

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(int value) : my_storage(value) {}

    /** Destructor */
    ~Atomic() = default;

protected:
    int my_storage;

private:
    Atomic(const Atomic &);
    Atomic & operator=(const Atomic &);
};


template <>
class DAAL_EXPORT Atomic<size_t>
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    inline size_t inc()
    {
        DAAL_ASSERT(sizeof(my_storage)==sizeof(size_t))
        return (size_t)(_InterlockedExchangeAdd64((__int64 *)(&my_storage), 1) + 1);
    }

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    inline size_t dec()
    {
        DAAL_ASSERT(sizeof(my_storage)==sizeof(size_t))
        return (size_t)(_InterlockedExchangeAdd64((__int64 *)(&my_storage), -1) - 1);
    }

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    inline void set(size_t value)
    {
        _ReadWriteBarrier();
        my_storage = value;
    }

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    inline size_t get() const
    {
        size_t to_return = my_storage;
        _ReadWriteBarrier();
        return to_return;
    }

    /**
     * Constructs an atomic object
     */
    Atomic() = default;

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(size_t value) : my_storage(value) {}

    /** Destructor */
    ~Atomic() = default;

protected:
    size_t my_storage;

private:
    Atomic(const Atomic &);
    Atomic & operator=(const Atomic &);
};
#endif // _WIN64

/** @} */

} // namespace interface1

using interface1::Atomic;

typedef Atomic<int> AtomicInt;
typedef Atomic<size_t> AtomicSizeT;

} // namespace services
} // namespace daal

#endif
