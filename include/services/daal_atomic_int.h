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

#include "tbb/tbb.h"
#include "tbb/atomic.h"
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
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
    dataType inc()
    {
        tbb::atomic<dataType> * atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
        return ++(*atomicPtr);
    }

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    dataType dec()
    {
        tbb::atomic<dataType> * atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
        return --(*atomicPtr);
    }

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    void set(dataType value)
    {
        tbb::atomic<dataType> * atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
        *atomicPtr                        = value;
    }

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    dataType get() const
    {
        tbb::atomic<dataType> * atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
        return *atomicPtr;
    }

    /**
     * Constructs an atomic object
     */
    Atomic()
    {
        this->_ptr = new tbb::atomic<dataType>();
    }

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(dataType value)
    {
        tbb::atomic<dataType> * atomicPtr = new tbb::atomic<dataType>();
        *atomicPtr                        = value;
        this->_ptr                        = atomicPtr;
    }

    /** Destructor */
    ~Atomic()
    {
        delete (tbb::atomic<dataType> *)(this->_ptr);
    }

protected:
    void * _ptr;

private:
    Atomic(const Atomic &);
    Atomic & operator=(const Atomic &);
};

/** @} */

} // namespace interface1
using interface1::Atomic;

typedef Atomic<int> AtomicInt;

} // namespace services
} // namespace daal

#endif
