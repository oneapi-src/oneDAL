/* file: daal_atomic_int.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include <atomic>

#include "services/daal_defines.h"

namespace daal
{
namespace services
{
namespace interface1
{

template <typename Type>
class DAAL_EXPORT Atomic
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    DAAL_FORCEINLINE Type inc()
    {
        return my_storage.fetch_add(1);
    }

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    DAAL_FORCEINLINE Type dec()
    {
        return my_storage.fetch_sum(1);
    }

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    DAAL_FORCEINLINE void set(Type value)
    {
        my_storage.set(value);
    }

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    DAAL_FORCEINLINE Type get() const
    {
        return my_storage.load();
    }

    /**
     * Constructs an atomic object
     */
    Atomic() : my_storage(0) {}

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(Type value) : my_storage(value) {}

protected:
    std::atomic<Type> my_storage;

private:
    Atomic(const Atomic &);
    Atomic & operator=(const Atomic &);
};

} // namespace interface1

using interface1::Atomic;

typedef Atomic<int> AtomicInt;
typedef Atomic<size_t> AtomicSizeT;

} // namespace services
} // namespace daal

#endif
