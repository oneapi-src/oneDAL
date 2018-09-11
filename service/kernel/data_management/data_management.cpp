/** file data_management.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

#include "collection.h"
#include "data_collection.h"
#include "memory_block.h"

namespace daal
{
namespace data_management
{

namespace interface1
{

DataCollection::DataCollection(size_t n) : super(n) {}

DataCollection::DataCollection() : super() {}

DataCollection::DataCollection(const DataCollection& other) : super(other) {}

const SerializationIfacePtr& DataCollection::operator[](size_t index) const
{
    return super::operator[](index);
}

SerializationIfacePtr& DataCollection::operator[](size_t index)
{
    return super::operator[](index);
}

SerializationIfacePtr &DataCollection::get(size_t index)
{
    return super::get(index);
}

const SerializationIfacePtr& DataCollection::get(size_t index) const
{
    return super::get(index);
}

DataCollection& DataCollection::push_back(const SerializationIfacePtr &x)
{
    super::push_back(x);
    return *this;
}

DataCollection& DataCollection::operator << (const SerializationIfacePtr &x)
{
    super::operator << (x);
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

MemoryBlock::MemoryBlock(size_t n): _size(n), _value(NULL)
{
    _value = (byte*)daal::services::daal_malloc(n);
}

MemoryBlock::~MemoryBlock()
{
    release();
}

void MemoryBlock::reserve(size_t n)
{
    if(n > size())
    {
        daal::services::daal_free(_value);
        _value = (byte*)daal::services::daal_malloc(n);
        _size = n;
    }
}

void MemoryBlock::release()
{
    if(_value)
    {
        daal::services::daal_free(_value);
        _value = NULL;
        _size = 0;
    }
}

}
}
}
