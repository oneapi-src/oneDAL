/* file: service_tensor.h */
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
//  Tensor service functions
//--
*/

#ifndef __SERVICE_TENSOR_H__
#define __SERVICE_TENSOR_H__

#include "homogen_tensor.h"
#include "service_defines.h"

using namespace daal::data_management;

namespace daal
{
namespace internal
{
template<typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode, typename TensorType>
class GetSubtensors
{
public:
    GetSubtensors(TensorType& data, size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum) : m_data(&data)
    {
        m_data->getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, mode, m_block);
    }
    GetSubtensors(TensorType *data, size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum) : m_data(data)
    {
        if(m_data)
            m_data->getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, mode, m_block);
    }
    GetSubtensors(TensorType& data, size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, const TensorOffsetLayout& layout) : m_data(&data)
    {
        if(m_data)
            m_data->getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, mode, m_block, layout);
    }
    GetSubtensors(TensorType *data, size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, const TensorOffsetLayout& layout) : m_data(data)
    {
        if(m_data)
            m_data->getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, mode, m_block, layout);
    }
    GetSubtensors() : m_data(nullptr){}
    ~GetSubtensors() { release(); }

    algorithmFPAccessType* get() { return m_data ? m_block.getPtr() : nullptr; }

    algorithmFPAccessType* next(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum)
    {
        if(!m_data)
            return nullptr;
        m_data->releaseSubtensor(m_block);
        m_data->getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, mode, m_block);
        return m_block.getPtr();
    }
    void set(TensorType& data, size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum)
    {
        release();
        m_data = &data;
        m_data->getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, mode, m_block);
    }
    void set(TensorType& data, size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, const TensorOffsetLayout& layout)
    {
        release();
        m_data = &data;
        m_data->getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, mode, m_block, layout);
    }
    void release()
    {
        if(m_data)
        {
            m_data->releaseSubtensor(m_block);
            m_data = nullptr;
        }
    }

private:
    TensorType* m_data;
    SubtensorDescriptor<algorithmFPType> m_block;
};

template<typename algorithmFPType, CpuType cpu, typename TensorType = Tensor>
using ReadSubtensor = GetSubtensors<algorithmFPType, const algorithmFPType, cpu, readOnly, TensorType>;

template<typename algorithmFPType, CpuType cpu, typename TensorType = Tensor>
using WriteSubtensor = GetSubtensors<algorithmFPType, algorithmFPType, cpu, readWrite, TensorType>;

template<typename algorithmFPType, CpuType cpu, typename TensorType = Tensor>
using WriteOnlySubtensor = GetSubtensors<algorithmFPType, algorithmFPType, cpu, writeOnly, TensorType>;

} // internal namespace
} // daal namespace

#endif
