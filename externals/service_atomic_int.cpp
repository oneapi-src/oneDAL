/* file: service_atomic_int.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of atomic functions.
//--
*/

#include "tbb/atomic.h"
#include "daal_atomic_int.h"

/*
//++
//  Implementation of Atomic<dataType> methods
//--
*/
template<typename dataType>
daal::services::Atomic<dataType>::Atomic() : _ptr(nullptr)
{
    this->_ptr = new tbb::atomic<dataType>();
}

template<typename dataType>
daal::services::Atomic<dataType>::Atomic(dataType value) : _ptr(nullptr)
{
    tbb::atomic<dataType> *atomicPtr = new tbb::atomic<dataType>();
    *atomicPtr = value;
    this->_ptr = atomicPtr;
}

template<typename dataType>
daal::services::Atomic<dataType>::~Atomic()
{
    delete (tbb::atomic<dataType> *)(this->_ptr);
}

template<typename dataType>
dataType daal::services::Atomic<dataType>::inc()
{
    tbb::atomic<dataType> *atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
    return ++(*atomicPtr);
}

template<typename dataType>
dataType daal::services::Atomic<dataType>::dec()
{
    tbb::atomic<dataType> *atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
    return --(*atomicPtr);
}

template<typename dataType>
void daal::services::Atomic<dataType>::set(dataType value)
{
    tbb::atomic<dataType> *atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
    *atomicPtr = value;
}

template<typename dataType>
dataType daal::services::Atomic<dataType>::get() const
{
    tbb::atomic<dataType> *atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
    return *atomicPtr;
}

/*
//++
//  Instantiation of Atomic classes
//--
*/
namespace daal
{
namespace services
{
namespace interface1
{
template class Atomic<int>;
template class Atomic<size_t>;
}
}
}
