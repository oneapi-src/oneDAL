/* file: service_tensor.h */
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
//  Tensor service functions
//--
*/

#ifndef __SERVICE_TENSOR_H__
#define __SERVICE_TENSOR_H__

#include "services/daal_memory.h"
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
    DAAL_NEW_DELETE();

    GetSubtensors(TensorType& data, size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum) : _data(&data)
    {
        getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum);
    }
    GetSubtensors(TensorType *data, size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum) : _data(data), _toReleaseFlag(false)
    {
        if(_data)
            getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum);
    }
    GetSubtensors(TensorType& data, size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, const TensorOffsetLayout& layout) : _data(&data)
    {
        getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, layout);
    }
    GetSubtensors(TensorType *data, size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, const TensorOffsetLayout& layout) : _data(data), _toReleaseFlag(false)
    {
        if(_data)
            getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, layout);
    }
    GetSubtensors(TensorType& data) : _data(&data)
    {
        getSubtensor(0, 0, 0, _data->getDimensionSize(0));
    }
    GetSubtensors(TensorType* data = nullptr) : _data(data), _toReleaseFlag(false)
    {
        if(_data)
            getSubtensor(0, 0, 0, _data->getDimensionSize(0));
    }
    ~GetSubtensors() { release(); }

    algorithmFPAccessType* get() { return _data ? _block.getPtr() : nullptr; }

    algorithmFPAccessType* next(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum)
    {
        if(!_data)
            return nullptr;
        if(_toReleaseFlag)
            _status = _data->releaseSubtensor(_block);
        return getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum);
    }
    algorithmFPAccessType* set(TensorType& data, size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum)
    {
        release();
        _data = &data;
        return getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum);
    }
    algorithmFPAccessType* set(TensorType *data, size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum)
    {
        release();

        if(data)
        {
            _data = data;
            return getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum);
        }
        return nullptr;
    }
    algorithmFPAccessType* set(TensorType& data, size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, const TensorOffsetLayout& layout)
    {
        release();
        _data = &data;
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, layout);
    }
    algorithmFPAccessType* set(TensorType *data, size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, const TensorOffsetLayout& layout)
    {
        release();
        if(data)
        {
            _data = data;
            return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, layout);
        }
        return nullptr;
    }
    void release()
    {
        if(_toReleaseFlag)
        {
            _data->releaseSubtensor(_block);
            _toReleaseFlag = false;
        }
        _data = nullptr;
        _status.clear();
    }
    size_t getSize()
    {
        return _block.getSize();
    }

    const services::Status& status() const { return _status; }

private:
    algorithmFPAccessType* getSubtensor(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum)
    {
        _status = _data->getSubtensor(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, mode, _block);
        _toReleaseFlag = _status.ok();
        return _block.getPtr();
    }

    algorithmFPAccessType* getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, const TensorOffsetLayout& layout)
    {
        _status = _data->getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, mode, _block, layout);
        _toReleaseFlag = _status.ok();
        return _block.getPtr();
    }

    TensorType* _data;
    SubtensorDescriptor<algorithmFPType> _block;
    services::Status _status;
    bool _toReleaseFlag;
};

template<typename algorithmFPType, CpuType cpu, typename TensorType = Tensor>
using ReadSubtensor = GetSubtensors<algorithmFPType, const algorithmFPType, cpu, readOnly, TensorType>;

template<typename algorithmFPType, CpuType cpu, typename TensorType = Tensor>
using WriteSubtensor = GetSubtensors<algorithmFPType, algorithmFPType, cpu, readWrite, TensorType>;

template<typename algorithmFPType, CpuType cpu, typename TensorType = Tensor>
using WriteOnlySubtensor = GetSubtensors<algorithmFPType, algorithmFPType, cpu, writeOnly, TensorType>;


/* Computes product of tensor dimensions from axisFrom (inclusive) up to axisTo (exclusive) */
size_t computeTensorDimensionsProd(const Tensor *tensor, size_t axisFrom, size_t axisTo);

/* Computes product of tensor dimensions before specified axis (exclusive) */
size_t computeTensorOffsetBeforeAxis(const Tensor *tensor, size_t axis);

/* Computes product of tensor dimensions after specified axis (exclusive) */
size_t computeTensorOffsetAfterAxis(const Tensor *tensor, size_t axis);

} // internal namespace
} // daal namespace

#endif
