/* file: homogen_tensor.h */
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
//  Declaration and implementation of the base class for numeric n-cubes.
//--
*/


#ifndef __HOMOGEN_TENSOR_H__
#define __HOMOGEN_TENSOR_H__

#include "services/daal_defines.h"
#include "data_management/data/tensor.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup tensor
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__HOMOGENTENSOR"></a>
 *  \brief Class that provides methods to access data stored as a contiguous array
 *  of homogeneous data in rows-major format.
 *  \tparam DataType Defines the underlying data type that describes a Tensor
 */
template<typename DataType = double>
class DAAL_EXPORT HomogenTensor : public Tensor
{
public:
    /** \private */
    HomogenTensor(size_t nDim, const size_t *dimSizes, DataType *data) : Tensor(&_layout), _layout(services::Collection<size_t>(nDim, dimSizes))
    {
        _ptr = data;
        _allocatedSize = 0;

        if( data )
        {
            _allocatedSize = getSize();
        }

        if(!dimSizes)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }
    }

    /** \private */
    HomogenTensor(const services::Collection<size_t> &dims, DataType *data);

    /** \private */
    HomogenTensor(const TensorOffsetLayout &layout, DataType *data) : Tensor(&_layout), _layout(layout)
    {
        const services::Collection<size_t>& dims = layout.getDimensions();
        _ptr = data;
        _allocatedSize = 0;

        if( data )
        {
            _allocatedSize = getSize();
        }

        size_t nDim = dims.size();

        if(nDim == 0)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }
    }

    /** \private */
    HomogenTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag) : Tensor(&_layout),
        _allocatedSize(0), _ptr(0), _layout(dims)
    {
        if( memoryAllocationFlag == doAllocate )
        {
            allocateDataMemory();
        }
    }

    /** \private */
    HomogenTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag, const DataType initValue):
        Tensor(&_layout), _allocatedSize(0), _ptr(0), _layout(dims)
    {
        if( memoryAllocationFlag == doAllocate )
        {
            allocateDataMemory();
            assign(initValue);
        }
    }

    /** \private */
    virtual ~HomogenTensor()
    {
        freeDataMemory();
    }

public:
    DataType *getArray() const
    {
        return _ptr;
    }

    void setArray( DataType *const ptr )
    {
        freeDataMemory();
        if(!ptr)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }
        _ptr = ptr;
        _memStatus = userAllocated;
    }

    TensorOffsetLayout& getTensorLayout()
    {
        return _layout;
    }

    virtual TensorOffsetLayout createDefaultSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        return TensorOffsetLayout(_layout);
    }

    virtual void setDimensions(size_t nDim, const size_t *dimSizes) DAAL_C11_OVERRIDE
    {
        if(!dimSizes)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        _layout = TensorOffsetLayout(services::Collection<size_t>(nDim, dimSizes));
    }

    virtual void setDimensions(const services::Collection<size_t>& dimensions) DAAL_C11_OVERRIDE
    {
        if(!dimensions.size())
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        _layout = TensorOffsetLayout(dimensions);
    }

    virtual void allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemory();

        if( _memStatus != notAllocated )
        {
            /* Error is already reported by freeDataMemory() */
            return;
        }

        size_t size = getSize();

        _ptr = (DataType *)daal::services::daal_malloc( size * sizeof(DataType) );

        if( _ptr == 0 )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        _allocatedSize = getSize();
        _memStatus = internallyAllocated;
    }

    void assign(const DataType initValue)
    {
        size_t size = getSize();

        for(size_t i = 0; i < size; i++)
        {
            _ptr[i] = initValue;
        }
    }

    virtual void freeDataMemory() DAAL_C11_OVERRIDE
    {
        if( getDataMemoryStatus() == internallyAllocated && _allocatedSize > 0 )
        {
            daal::services::daal_free(_ptr);
        }

        _ptr = 0;
        _allocatedSize = 0;
        _memStatus = notAllocated;
    }

    void getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        ReadWriteMode rwflag, SubtensorDescriptor<double> &block,
                        const TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;
    void getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        ReadWriteMode rwflag, SubtensorDescriptor<float> &block,
                        const TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;
    void getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        ReadWriteMode rwflag, SubtensorDescriptor<int> &block,
                        const TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;

    void releaseSubtensor(SubtensorDescriptor<double> &block) DAAL_C11_OVERRIDE;
    void releaseSubtensor(SubtensorDescriptor<float>  &block) DAAL_C11_OVERRIDE;
    void releaseSubtensor(SubtensorDescriptor<int>    &block) DAAL_C11_OVERRIDE;

    virtual services::SharedPtr<Tensor> getSampleTensor(size_t firstDimIndex) DAAL_C11_OVERRIDE
    {
        services::Collection<size_t> newDims = getDimensions();
        if(!_ptr || newDims.size() == 0 || newDims[0] <= firstDimIndex) { return services::SharedPtr<Tensor>(); }
        newDims[0] = 1;
        const size_t *_dimOffsets = &((_layout.getOffsets())[0]);
        return services::SharedPtr<Tensor>(new HomogenTensor<DataType>(newDims, _ptr + _dimOffsets[0]*firstDimIndex));
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return data_feature_utils::getIndexNumType<DataType>() + SERIALIZATION_HOMOGEN_TENSOR_ID;
    }

    void serializeImpl  (InputDataArchive  *archive) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( archive );}

    void deserializeImpl(OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( archive );}

protected:
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *archive ) {}

private:
    template <typename T>
    void getTSubtensor( size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, int rwFlag,
                        SubtensorDescriptor<T> &block, const TensorOffsetLayout& layout );
    template <typename T>
    void releaseTSubtensor( SubtensorDescriptor<T> &block );

private:
    DataType *_ptr;
    size_t    _allocatedSize;
    TensorOffsetLayout _layout;
};
/** @} */

}
using interface1::HomogenTensor;

}
} // namespace daal

#endif
