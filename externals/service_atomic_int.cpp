/* file: service_atomic_int.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
template <typename dataType>
daal::services::Atomic<dataType>::Atomic() : _ptr(nullptr)
{
    this->_ptr = new tbb::atomic<dataType>();
}

template <typename dataType>
daal::services::Atomic<dataType>::Atomic(dataType value) : _ptr(nullptr)
{
    tbb::atomic<dataType> * atomicPtr = new tbb::atomic<dataType>();
    *atomicPtr                        = value;
    this->_ptr                        = atomicPtr;
}

template <typename dataType>
daal::services::Atomic<dataType>::~Atomic()
{
    delete (tbb::atomic<dataType> *)(this->_ptr);
}

template <typename dataType>
dataType daal::services::Atomic<dataType>::inc()
{
    tbb::atomic<dataType> * atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
    return ++(*atomicPtr);
}

template <typename dataType>
dataType daal::services::Atomic<dataType>::dec()
{
    tbb::atomic<dataType> * atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
    return --(*atomicPtr);
}

template <typename dataType>
void daal::services::Atomic<dataType>::set(dataType value)
{
    tbb::atomic<dataType> * atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
    *atomicPtr                        = value;
}

template <typename dataType>
dataType daal::services::Atomic<dataType>::get() const
{
    tbb::atomic<dataType> * atomicPtr = (tbb::atomic<dataType> *)(this->_ptr);
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
} // namespace interface1
} // namespace services
} // namespace daal
