/** file data_management.cpp */
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

#include "services/collection.h"
#include "data_management/data/data_collection.h"
#include "data_management/data/memory_block.h"

namespace daal
{
namespace data_management
{
DataCollection::DataCollection(size_t n) : super(n) {}

DataCollection::DataCollection() : super() {}

DataCollection::DataCollection(const DataCollection & other) : super(other) {}

const SerializationIfacePtr & DataCollection::operator[](size_t index) const
{
    return super::operator[](index);
}

SerializationIfacePtr & DataCollection::operator[](size_t index)
{
    return super::operator[](index);
}

SerializationIfacePtr & DataCollection::get(size_t index)
{
    return super::get(index);
}

const SerializationIfacePtr & DataCollection::get(size_t index) const
{
    return super::get(index);
}

DataCollection & DataCollection::push_back(const SerializationIfacePtr & x)
{
    super::push_back(x);
    return *this;
}

DataCollection & DataCollection::operator<<(const SerializationIfacePtr & x)
{
    super::operator<<(x);
    return *this;
}

size_t DataCollection::size() const
{
    return super::size();
}

void DataCollection::clear()
{
    super::clear();
}

void DataCollection::erase(size_t pos)
{
    super::erase(pos);
}

bool DataCollection::resize(size_t newCapacity)
{
    return super::resize(newCapacity);
}

MemoryBlock::MemoryBlock(size_t n) : _size(n), _value(NULL)
{
    _value = (byte *)daal::services::daal_calloc(n);
}

MemoryBlock::~MemoryBlock()
{
    release();
}

void MemoryBlock::reserve(size_t n)
{
    if (n > size())
    {
        daal::services::daal_free(_value);
        _value = (byte *)daal::services::daal_calloc(n);
        _size  = n;
    }
}

void MemoryBlock::release()
{
    if (_value)
    {
        daal::services::daal_free(_value);
        _value = NULL;
        _size  = 0;
    }
}

} // namespace data_management
} // namespace daal
