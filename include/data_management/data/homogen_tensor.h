/* file: homogen_tensor.h */
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

/*
//++
//  Declaration and implementation of the base class for numeric n-cubes.
//--
*/


#ifndef __HOMOGEN_TENSOR_H__
#define __HOMOGEN_TENSOR_H__

#include "services/daal_defines.h"
#include "data_management/data/tensor.h"
#include "data_management/data/data_serialize.h"

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
 *  \tparam DataType Defines the underlying data type that describes a tensor
 */
template<typename DataType = DAAL_DATA_TYPE>
class DAAL_EXPORT HomogenTensor : public Tensor
{
public:
    DECLARE_SERIALIZABLE_TAG();
    DECLARE_SERIALIZABLE_IMPL();

    DAAL_CAST_OPERATOR(HomogenTensor<DataType>);

    /**
     * Constructs an empty homogeneous tensor
     * \DAAL_DEPRECATED_USE{ HomogenTensor::create }
     */
    HomogenTensor() :
        Tensor(&_layout), _allocatedSize(0), _layout(services::Collection<size_t>(0, 0))
    {}

    /**
     * Constructs an empty homogeneous tensor
     * \param[out] stat Status of the tensor construction
     * \return Empty homogeneous tensor
     */
    static services::SharedPtr<HomogenTensor<DataType> > create(services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL(HomogenTensor, DataType);
    }

    /**
     * Constructs homogeneous tensor with a user-defined data, number and sizes of dimensions
     * \param nDim     Number of dimensions
     * \param dimSizes Array with sizes for each dimension
     * \param data     Array with a homogeneous data set
     * \DAAL_DEPRECATED_USE{ HomogenTensor::create }
     */
    HomogenTensor(size_t nDim, const size_t *dimSizes, DataType *data) : Tensor(&_layout), _layout(services::Collection<size_t>(nDim, dimSizes))
    {
        _ptr = services::SharedPtr<byte>((byte*)data, services::EmptyDeleter());
        _allocatedSize = 0;

        if( data )
        {
            _allocatedSize = getSize();
            _memStatus = userAllocated;
        }

        if(!dimSizes)
        {
            this->_status |= services::ErrorNullParameterNotSupported;
            return;
        }
    }

    /**
     * Constructs homogeneous tensor with a user-defined data, number and sizes of dimensions
     * \param[in] nDim     Number of dimensions
     * \param[in] dimSizes Array with sizes for each dimension
     * \param[in] data     Array with a homogeneous data set
     * \param[out] stat    Status of the tensor construction
     * \return Homogeneous tensor with a user-defined data, number and sizes of dimensions
     */
    static services::SharedPtr<HomogenTensor<DataType> > create(size_t nDim, const size_t *dimSizes, DataType *data,
                                                                services::Status *stat = NULL)
    {
        return create(nDim, dimSizes, services::SharedPtr<DataType>(data, services::EmptyDeleter()), stat);
    }

    /**
     * Constructs homogeneous tensor with a user-defined data, number and sizes of dimensions
     * \param nDim     Number of dimensions
     * \param dimSizes Array with sizes for each dimension
     * \param data     Array with a homogeneous data set
     * \DAAL_DEPRECATED_USE{ HomogenTensor::create }
     */
    HomogenTensor(size_t nDim, const size_t *dimSizes, const services::SharedPtr<DataType>& data) : Tensor(&_layout), _layout(services::Collection<size_t>(nDim, dimSizes))
    {
        _ptr = services::reinterpretPointerCast<byte, DataType>(data);
        _allocatedSize = 0;

        if( data )
        {
            _allocatedSize = getSize();
            _memStatus = userAllocated;
        }

        if(!dimSizes)
        {
            this->_status |= services::ErrorNullParameterNotSupported;
            return;
        }
    }

    /**
     * Constructs homogeneous tensor with a user-defined data, number and sizes of dimensions
     * \param[in] nDim     Number of dimensions
     * \param[in] dimSizes Array with sizes for each dimension
     * \param[in] data     Array with a homogeneous data set
     * \param[out] stat    Status of the tensor construction
     * \return Homogeneous tensor with a user-defined data, number and sizes of dimensions
     */
    static services::SharedPtr<HomogenTensor<DataType> > create(size_t nDim, const size_t *dimSizes,
                                                                const services::SharedPtr<DataType>& data,
                                                                services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenTensor, DataType, nDim, dimSizes, data);
    }

    /**
     * Constructs homogeneous tensor with a user-defined data and dimensions
     * \param dims  Collection of tensor dimensions
     * \param data  Size of the byte array
     * \DAAL_DEPRECATED_USE{ HomogenTensor::create }
     */
    HomogenTensor(const services::Collection<size_t> &dims, DataType *data);

    /**
     * Constructs homogeneous tensor with a user-defined data, number and sizes of dimensions
     * \param[in] dims     Collection of tensor dimensions
     * \param[in] data     Array with a homogeneous data set
     * \param[out] stat    Status of the tensor construction
     * \return Homogeneous tensor with a user-defined data, number and sizes of dimensions
     */
    static services::SharedPtr<HomogenTensor<DataType> > create(const services::Collection<size_t> &dims, DataType *data,
                                                                services::Status *stat = NULL)
    {
        return create(dims, services::SharedPtr<DataType>(data, services::EmptyDeleter()), stat);
    }

    /**
     * Constructs homogeneous tensor with a user-defined data and dimensions
     * \param dims  Collection of tensor dimensions
     * \param data  Array with a homogeneous data set
     * \DAAL_DEPRECATED_USE{ HomogenTensor::create }
     */
    HomogenTensor(const services::Collection<size_t> &dims, const services::SharedPtr<DataType> &data);

    /**
     * Constructs homogeneous tensor with a user-defined data, number and sizes of dimensions
     * \param[in] dims     Collection of tensor dimensions
     * \param[in] data     Array with a homogeneous data set
     * \param[out] stat    Status of the tensor construction
     * \return Homogeneous tensor with a user-defined data, number and sizes of dimensions
     */
    static services::SharedPtr<HomogenTensor<DataType> > create(const services::Collection<size_t> &dims,
                                                                const services::SharedPtr<DataType> &data,
                                                                services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenTensor, DataType, dims, data);
    }

    /**
     * Constructs homogeneous tensor with a user-defined data and layout
     * \param[in] layout   Tensor layout
     * \param[in] data     Array with a homogeneous data set
     * \DAAL_DEPRECATED_USE{ HomogenTensor::create }
     */
    HomogenTensor(const TensorOffsetLayout &layout, DataType *data) : Tensor(&_layout), _layout(layout)
    {
        const services::Collection<size_t>& dims = layout.getDimensions();
        _ptr = services::SharedPtr<byte>((byte*)data, services::EmptyDeleter());
        _allocatedSize = 0;

        if( data )
        {
            _allocatedSize = getSize();
            _memStatus = userAllocated;
        }

        size_t nDim = dims.size();

        if(nDim == 0)
        {
            this->_status |= services::ErrorNullParameterNotSupported;
            return;
        }
    }

    /**
     * Constructs homogeneous tensor with a user-defined data and layout
     * \param[in] layout   Tensor layout
     * \param[in] data     Array with a homogeneous data set
     * \param[out] stat    Status of the tensor construction
     * \return Homogeneous tensor with a user-defined data and layout
     */
    static services::SharedPtr<HomogenTensor<DataType> > create(const TensorOffsetLayout &layout, DataType *data,
                                                                services::Status *stat = NULL)
    {
        return create(layout, services::SharedPtr<DataType>(data, services::EmptyDeleter()), stat);
    }

    /**
     * Constructs homogeneous tensor with a user-defined data and layout
     * \param[in] layout   Tensor layout
     * \param[in] data     Array with a homogeneous data set
     * \DAAL_DEPRECATED_USE{ HomogenTensor::create }
     */
    HomogenTensor(const TensorOffsetLayout &layout, const services::SharedPtr<DataType> &data) : Tensor(&_layout), _layout(layout)
    {
        const services::Collection<size_t>& dims = layout.getDimensions();
        _ptr = services::reinterpretPointerCast<byte, DataType>(data);
        _allocatedSize = 0;

        if( data )
        {
            _allocatedSize = getSize();
            _memStatus = userAllocated;
        }

        size_t nDim = dims.size();

        if(nDim == 0)
        {
            this->_status |= services::ErrorNullParameterNotSupported;
            return;
        }
    }

    /**
     * Constructs homogeneous tensor with a user-defined data and layout
     * \param[in] layout   Tensor layout
     * \param[in] data     Array with a homogeneous data set
     * \param[out] stat    Status of the tensor construction
     * \return Homogeneous tensor with a user-defined data and layout
     */
    static services::SharedPtr<HomogenTensor<DataType> > create(const TensorOffsetLayout &layout, const services::SharedPtr<DataType> &data,
                                                                services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenTensor, DataType, layout, data);
    }

    /**
     * Constructs homogeneous tensor with a user-defined dimensions and memory allocation flag
     * \param dims                  Collection of tensor dimensions
     * \param memoryAllocationFlag  Flag specifying memory allocation
     * \DAAL_DEPRECATED_USE{ HomogenTensor::create }
     */
    HomogenTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag) : Tensor(&_layout),
        _allocatedSize(0), _layout(dims)
    {
        if( memoryAllocationFlag == doAllocate )
        {
            this->_status |= allocateDataMemoryImpl();
        }
    }

    /**
     * Constructs homogeneous tensor with a user-defined dimensions and memory allocation flag
     * \param[in] dims                  Collection of tensor dimensions
     * \param[in] memoryAllocationFlag  Flag specifying memory allocation
     * \param[out] stat                 Status of the tensor construction
     * \return Homogeneous tensor with a user-defined dimensions
     */
    static services::SharedPtr<HomogenTensor<DataType> > create(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag,
                                                                services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenTensor, DataType, dims, memoryAllocationFlag);
    }

    /**
     * Constructs homogeneous tensor with a user-defined dimensions, memory allocation flag and fills it with the constant values
     * \param dims                  Collection of tensor dimensions
     * \param memoryAllocationFlag  Flag specifying memory allocation
     * \param initValue             Constant value
     * \DAAL_DEPRECATED_USE{ HomogenTensor::create }
     */
    HomogenTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag, const DataType initValue):
        Tensor(&_layout), _allocatedSize(0), _layout(dims)
    {
        if( memoryAllocationFlag == doAllocate )
        {
            this->_status |= allocateDataMemoryImpl();
            this->_status |= assign(initValue);
        }
    }

    /**
     * Constructs homogeneous tensor with a user-defined dimensions and memory allocation flag and fills it with the constant values
     * \param[in] dims                  Collection of tensor dimensions
     * \param[in] memoryAllocationFlag  Flag specifying memory allocation
     * \param[in] initValue             Constant value
     * \param[out] stat                 Status of the tensor construction
     * \return Homogeneous tensor with a user-defined dimensions
     */
    static services::SharedPtr<HomogenTensor<DataType> > create(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag,
                                                                const DataType initValue, services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenTensor, DataType, dims, memoryAllocationFlag, initValue);
    }

    /** \private */
    virtual ~HomogenTensor()
    {
        freeDataMemoryImpl();
    }

public:
    /**
     *  Returns a pointer to a data set registered in a homogeneous tensor
     *  \return Pointer to the data set
     */
    DataType *getArray() const
    {
        return (DataType*)_ptr.get();
    }

    /**
     *  Returns a pointer to a data set registered in a homogeneous tensor
     *  \return Pointer to the data set
     */
    services::SharedPtr<DataType> getArraySharedPtr() const
    {
        return services::reinterpretPointerCast<DataType, byte>(_ptr);
    }

    /**
     *  Sets a pointer to a homogeneous data set
     *  \param[in] ptr Pointer to the data set in the homogeneous format
     */
    services::Status setArray( DataType *const ptr )
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, freeDataMemoryImpl());
        if(!ptr)
        {
            return services::Status(services::ErrorNullParameterNotSupported);
        }
        _ptr = services::SharedPtr<byte>((byte*)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;
        return s;
    }

    /**
     *  Sets a pointer to a homogeneous data set
     *  \param[in] ptr Pointer to the data set in the homogeneous format
     */
    services::Status setArray( const services::SharedPtr<DataType>& ptr )
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, freeDataMemoryImpl());
        if(!ptr)
        {
            return services::Status(services::ErrorNullParameterNotSupported);
        }
        _ptr = ptr;
        _memStatus = userAllocated;
        return s;
    }

    /**
     *  Returns a homogeneous tensor offset layout
     *  \return Layout
     */
    TensorOffsetLayout& getTensorLayout()
    {
        return _layout;
    }

    /**
     *  Creates default tensor layout
     *  \return Layout
     */
    virtual TensorOffsetLayout createDefaultSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        return TensorOffsetLayout(_layout);
    }

    /**
     *  Creates raw tensor layout
     *  \return Layout
     */
    virtual TensorOffsetLayout createRawSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        TensorOffsetLayout layout(_layout);
        layout.sortOffsets();
        return layout;
    }

    //the descriptions of the methods below are inherited from the base class
    virtual services::Status setDimensions(size_t nDim, const size_t *dimSizes) DAAL_C11_OVERRIDE
    {
        if(!dimSizes)
        {
            return services::Status(services::ErrorNullParameterNotSupported);
        }

        _layout = TensorOffsetLayout(services::Collection<size_t>(nDim, dimSizes));
        return services::Status();
    }

    virtual services::Status setDimensions(const services::Collection<size_t>& dimensions) DAAL_C11_OVERRIDE
    {
        if(!dimensions.size())
        {
            return services::Status(services::ErrorNullParameterNotSupported);
        }

        _layout = TensorOffsetLayout(dimensions);
        return services::Status();
    }

    /**
     *  Fills a tensor with a constant value
     *  \param[in]  initValue  Constant value to initialize entries of the homogeneous tensor
     */
    services::Status assign(const DataType initValue)
    {
        size_t size = getSize();

        DataType * ptr = (DataType*)_ptr.get();
        for(size_t i = 0; i < size; i++)
        {
            ptr[i] = initValue;
        }
        return services::Status();
    }

    services::Status getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        ReadWriteMode rwflag, SubtensorDescriptor<double> &block,
                        const TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;
    services::Status getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        ReadWriteMode rwflag, SubtensorDescriptor<float> &block,
                        const TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;
    services::Status getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        ReadWriteMode rwflag, SubtensorDescriptor<int> &block,
                        const TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;

    services::Status getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<double>& subtensor ) DAAL_C11_OVERRIDE
    {
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, _layout );
    }

    services::Status getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<float>& subtensor ) DAAL_C11_OVERRIDE
    {
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, _layout );
    }

    services::Status getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<int>& subtensor ) DAAL_C11_OVERRIDE
    {
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, _layout );
    }

    services::Status releaseSubtensor(SubtensorDescriptor<double> &block) DAAL_C11_OVERRIDE;
    services::Status releaseSubtensor(SubtensorDescriptor<float>  &block) DAAL_C11_OVERRIDE;
    services::Status releaseSubtensor(SubtensorDescriptor<int>    &block) DAAL_C11_OVERRIDE;

    DAAL_DEPRECATED_VIRTUAL virtual services::SharedPtr<Tensor> getSampleTensor(size_t firstDimIndex) DAAL_C11_OVERRIDE
    {
        services::Collection<size_t> newDims = getDimensions();
        if(!_ptr || newDims.size() == 0 || newDims[0] <= firstDimIndex) { return services::SharedPtr<Tensor>(); }
        newDims[0] = 1;
        const size_t *_dimOffsets = &((_layout.getOffsets())[0]);
        services::SharedPtr<DataType> shiftedPtr(services::reinterpretPointerCast<DataType, byte>(_ptr), (DataType*)_ptr.get() + _dimOffsets[0] * firstDimIndex);
        services::Status st;
        return HomogenTensor<DataType>::create(newDims, shiftedPtr, &st);
    }

protected:

    HomogenTensor(services::Status &st) :
        Tensor(&_layout, st), _allocatedSize(0), _layout(services::Collection<size_t>(0, 0))
    {}

    HomogenTensor(size_t nDim, const size_t *dimSizes, const services::SharedPtr<DataType>& data, services::Status &st) :
        Tensor(&_layout, st), _layout(services::Collection<size_t>(nDim, dimSizes))
    {
        _ptr = services::reinterpretPointerCast<byte, DataType>(data);
        _allocatedSize = 0;

        if( data )
        {
            _allocatedSize = getSize();
            _memStatus = userAllocated;
        }

        if(!dimSizes)
        {
            st |= services::ErrorNullParameterNotSupported;
            return;
        }
    }

    HomogenTensor(const services::Collection<size_t> &dims, const services::SharedPtr<DataType> &data, services::Status &st);

    HomogenTensor(const TensorOffsetLayout &layout, const services::SharedPtr<DataType> &data,
                  services::Status &st) :
        Tensor(&_layout, st), _layout(layout)
    {
        const services::Collection<size_t>& dims = layout.getDimensions();
        _ptr = services::reinterpretPointerCast<byte, DataType>(data);
        _allocatedSize = 0;

        if(data)
        {
            _allocatedSize = getSize();
            _memStatus = userAllocated;
        }

        size_t nDim = dims.size();

        if(nDim == 0)
        {
            st |= services::ErrorNullParameterNotSupported;
            return;
        }
    }

    HomogenTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag,
                  services::Status &st) :
        Tensor(&_layout, st), _allocatedSize(0), _layout(dims)
    {
        if( memoryAllocationFlag == doAllocate )
        {
            st |= allocateDataMemoryImpl();
        }
    }

    HomogenTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag, const DataType initValue,
                  services::Status &st):
        Tensor(&_layout, st), _allocatedSize(0), _layout(dims)
    {
        if(memoryAllocationFlag == doAllocate)
        {
            st |= allocateDataMemoryImpl();
            if (!st)
                return;
            st |= assign(initValue);
        }
    }

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl( Archive *archive )
    {
        Tensor::serialImpl<Archive, onDeserialize>( archive );

        archive->setObj( &_layout );

        bool isAllocated = (_memStatus != notAllocated);
        archive->set( isAllocated );

        if( onDeserialize )
        {
            freeDataMemoryImpl();

            if( isAllocated )
            {
                allocateDataMemoryImpl();
            }
        }

        if(_memStatus != notAllocated)
        {
            archive->set( (DataType*)_ptr.get(), getSize() );
        }

        return services::Status();
    }

    virtual services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, freeDataMemoryImpl());

        size_t size = getSize();

        if( size == 0 )
        {
            return services::Status();
        }

        _ptr = services::SharedPtr<byte>((byte *)daal::services::daal_malloc(size * sizeof(DataType)), services::ServiceDeleter());

        if( _ptr == 0 )
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        _allocatedSize = getSize();
        _memStatus = internallyAllocated;
        return services::Status();
    }

    //the descriptions of the methods below are inherited from the base class
    virtual services::Status freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        if(_ptr)
            _ptr = services::SharedPtr<byte>();
        _allocatedSize = 0;
        _memStatus = notAllocated;
        return services::Status();
    }

private:
    template <typename T>
    services::Status getTSubtensor( size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, int rwFlag,
                        SubtensorDescriptor<T> &block, const TensorOffsetLayout& layout );
    template <typename T>
    services::Status releaseTSubtensor( SubtensorDescriptor<T> &block );

private:
    services::SharedPtr<byte> _ptr;
    size_t _allocatedSize;
    TensorOffsetLayout _layout;
};
/** @} */

}
using interface1::HomogenTensor;

}
} // namespace daal

#endif
