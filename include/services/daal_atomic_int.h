/* file: daal_atomic_int.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "services/daal_defines.h"
#include "services/daal_memory.h"

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
template<typename dataType = int>
class DAAL_EXPORT Atomic
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    int inc();

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    int dec();

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    void set(dataType value);

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    dataType get() const;

    /**
     * Constructs an atomic object
     */
    Atomic();

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(dataType value);

    /** Destructor */
    virtual ~Atomic();

protected:
    void *_ptr;
};

/**
 * <a name="DAAL-CLASS-SERVICES__ATOMIC_INT"></a>
 * \brief Class that represents an atomic integer object
 */
template<>
class DAAL_EXPORT Atomic<int>
{
public:
    DAAL_NEW_DELETE();

    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    int inc();

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    int dec();

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    void set(int value);

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    int get() const;

    /**
     * Constructs an atomic object
     */
    Atomic();

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(int value);

    /** Destructor */
    virtual ~Atomic();

protected:
    void *_ptr;
};

/**
 * <a name="DAAL-CLASS-SERVICES__ATOMIC_SIZE_T"></a>
 * \brief Class that represents an atomic unsigned integer object
 */
template<>
class DAAL_EXPORT Atomic<size_t>
{
public:
    DAAL_NEW_DELETE();

    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    size_t inc();

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    size_t dec();

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    void set(size_t value);

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    size_t get() const;

    /**
     * Constructs an atomic object
     */
    Atomic();

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(size_t value);

    /** Destructor */
    virtual ~Atomic();

protected:
    void *_ptr;
};
/** @} */

} // namespace interface1
using interface1::Atomic;

typedef Atomic<int> AtomicInt;

}
}

#endif
