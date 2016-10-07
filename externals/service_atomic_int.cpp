/* file: service_atomic_int.cpp */
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
//  Implementation of atomic functions.
//--
*/

#include "tbb/atomic.h"
#include "daal_atomic_int.h"

/*
//++
//  Implementation of Atomic<int> methods
//--
*/
daal::services::Atomic<int>::Atomic() : _ptr(nullptr)
{
    this->_ptr = new tbb::atomic<int>();
}

daal::services::Atomic<int>::Atomic(int value) : _ptr(nullptr)
{
    tbb::atomic<int> *atomicPtr = new tbb::atomic<int>();
    *atomicPtr = value;
    this->_ptr = atomicPtr;
}

daal::services::Atomic<int>::~Atomic()
{
    delete (tbb::atomic<int>*)(this->_ptr);
}

int daal::services::Atomic<int>::inc()
{
    tbb::atomic<int> *atomicPtr = (tbb::atomic<int>*)(this->_ptr);
    return ++(*atomicPtr);
}

int daal::services::Atomic<int>::dec()
{
    tbb::atomic<int> *atomicPtr = (tbb::atomic<int>*)(this->_ptr);
    return --(*atomicPtr);
}

void daal::services::Atomic<int>::set(int value)
{
    tbb::atomic<int> *atomicPtr = (tbb::atomic<int>*)(this->_ptr);
    *atomicPtr = value;
}

int daal::services::Atomic<int>::get() const
{
    tbb::atomic<int> *atomicPtr = (tbb::atomic<int>*)(this->_ptr);
    return *atomicPtr;
}

/*
//++
//  Implementation of Atomic<size_t> methods
//--
*/
daal::services::Atomic<size_t>::Atomic() : _ptr(nullptr)
{
    this->_ptr = new tbb::atomic<size_t>();
}

daal::services::Atomic<size_t>::Atomic(size_t value) : _ptr(nullptr)
{
    tbb::atomic<size_t> *atomicPtr = new tbb::atomic<size_t>();
    *atomicPtr = value;
    this->_ptr = atomicPtr;
}

daal::services::Atomic<size_t>::~Atomic()
{
    delete (tbb::atomic<size_t>*)(this->_ptr);
}

size_t daal::services::Atomic<size_t>::inc()
{
    tbb::atomic<size_t> *atomicPtr = (tbb::atomic<size_t>*)(this->_ptr);
    return ++(*atomicPtr);
}

size_t daal::services::Atomic<size_t>::dec()
{
    tbb::atomic<size_t> *atomicPtr = (tbb::atomic<size_t>*)(this->_ptr);
    return --(*atomicPtr);
}

void daal::services::Atomic<size_t>::set(size_t value)
{
    tbb::atomic<size_t> *atomicPtr = (tbb::atomic<size_t>*)(this->_ptr);
    *atomicPtr = value;
}

size_t daal::services::Atomic<size_t>::get() const
{
    tbb::atomic<size_t> *atomicPtr = (tbb::atomic<size_t>*)(this->_ptr);
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
