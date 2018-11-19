/* file: service_mkl_tensor.h */
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


#ifndef __MKL_TENSOR_H__
#define __MKL_TENSOR_H__

#include "services/daal_defines.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/data_serialize.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
namespace internal
{

template<typename FPType>
class DnnLayoutDeleter : public services::DeleterIface
{
public:
    void operator() (const void *ptr) DAAL_C11_OVERRIDE;
};

class DnnLayoutPtr
{
public:
    DAAL_NEW_DELETE();

    DnnLayoutPtr() : _ownedPtr(NULL), _ptr(NULL), _refCount(NULL)
    {
    }

    template<class D>
    explicit DnnLayoutPtr(void *ptr, const D& deleter) : _ownedPtr(ptr), _ptr(ptr), _refCount(NULL)
    {
        if(_ownedPtr)
            _refCount = new services::RefCounterImp<D>(deleter);
    }

    DnnLayoutPtr(const DnnLayoutPtr &other);

    ~DnnLayoutPtr() { _remove(); }

    DnnLayoutPtr &operator=(const DnnLayoutPtr &ptr);

    void reset()
    {
        _remove();
        _ownedPtr = NULL;
        _refCount = NULL;
        _ptr = NULL;
    }

    void *operator->() const { return _ptr; }

    operator bool() const { return (_ptr != NULL); }

    void *get() const { return _ptr; }

    void *getStartPtr() const { return _ownedPtr; }

    int useCount() const { return _refCount ? _refCount->get() : 0; }

protected:
    void *_ownedPtr;           /* Pointer to the beginning of the owned memory */
    void *_ptr;                /* Pointer to return */
    services::RefCounter *_refCount;  /* Reference count */

    /**
    * Decreases the reference count
    * If the reference count becomes equal to zero, deletes the owned pointer
    */
    void _remove();
}; // class DnnLayoutPtr

template<typename DataType = double>
class DAAL_EXPORT MklTensor : public data_management::Tensor
{
public:
    DECLARE_SERIALIZABLE_TAG();

    DAAL_CAST_OPERATOR(MklTensor<DataType>)

    /** \private */
    MklTensor() : data_management::Tensor(&_layout), _layout(services::Collection<size_t>()),
        _dnnPtr(NULL), _isDnnLayout(false),
        _plainPtr(NULL), _isPlainLayout(false)
    {
    }

    MklTensor(size_t nDim, const size_t *dimSizes);

    MklTensor(size_t nDim, const size_t *dimSizes, AllocationFlag memoryAllocationFlag);

    MklTensor(const services::Collection<size_t> &dims);

    MklTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag);

    static services::SharedPtr<MklTensor<DataType> > create(services::Status *stat)
    {
        DAAL_DEFAULT_CREATE_IMPL(MklTensor<DataType>);
    }

    static services::SharedPtr<MklTensor<DataType> > create(size_t nDim, const size_t *dimSizes, services::Status *stat)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(MklTensor<DataType>, nDim, dimSizes);
    }

    static services::SharedPtr<MklTensor<DataType> > create(size_t nDim, const size_t *dimSizes,
                                                            AllocationFlag memoryAllocationFlag, services::Status *stat)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(MklTensor<DataType>, nDim, dimSizes, memoryAllocationFlag);
    }

    static services::SharedPtr<MklTensor<DataType> > create(const services::Collection<size_t> &dims, services::Status *stat)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(MklTensor<DataType>, dims);
    }

    static services::SharedPtr<MklTensor<DataType> > create(const services::Collection<size_t> &dims,
                                                            AllocationFlag memoryAllocationFlag, services::Status *stat)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(MklTensor<DataType>, dims, memoryAllocationFlag);
    }

    static services::SharedPtr<MklTensor<DataType> > createWithDnnLayout(const services::Collection<size_t> &dims, void *layout,
                                                                         AllocationFlag memoryAllocationFlag, services::Status *stat)
    {
        auto tensor = MklTensor<DataType>::create(dims, memoryAllocationFlag, stat);
        if (!stat->ok()) { return services::SharedPtr<MklTensor<DataType> >(); }

        stat->add(tensor->setDnnLayout(layout));
        return tensor;
    }

    /**
     * MklTensor copying isn't implemented
     */
    MklTensor(const MklTensor &other) = delete;
    MklTensor &operator=(const MklTensor &other) = delete;

    /** \private */
    virtual ~MklTensor()
    {
        freeDataMemory();
        freeDnnLayout();
        freePlainLayout();
    }

    DataType* getDnnArray()
    {
        if (_dnnLayout)
        {
            syncPlainToDnn();
            _isPlainLayout = false;
            return _dnnPtr;
        }

        return _plainPtr;
    }

    DnnLayoutPtr getSharedDnnLayout()
    {
        if (_dnnLayout)
        {
            return _dnnLayout;
        }

        return _plainLayout;
    }

    void* getDnnLayout()
    {
        if (_dnnLayout)
        {
            return _dnnLayout.get();
        }

        return _plainLayout.get();
    }

    services::Status setDnnLayout(const DnnLayoutPtr &dnnLayout);

    services::Status setDnnLayout(void* dnnLayout);

    services::Status syncDnnToPlain();

    DataType* getPlainArray()
    {
        syncDnnToPlain();
        _isDnnLayout = false;
        return _plainPtr;
    }

    bool isPlainLayout()
    {
        return _isPlainLayout;
    }

    bool isDnnLayout()
    {
        return _isDnnLayout;
    }

    data_management::TensorOffsetLayout& getTensorLayout()
    {
        return _layout;
    }

    virtual data_management::TensorOffsetLayout createDefaultSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        return data_management::TensorOffsetLayout(_layout);
    }

    virtual data_management::TensorOffsetLayout createRawSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        data_management::TensorOffsetLayout layout(_layout);
        layout.sortOffsets();
        return layout;
    }

    virtual services::Status setDimensions(size_t nDim, const size_t *dimSizes) DAAL_C11_OVERRIDE
    {
        if(!dimSizes)
        {
            return services::Status(services::ErrorNullParameterNotSupported);
        }

        _layout = data_management::TensorOffsetLayout(services::Collection<size_t>(nDim, dimSizes));
        return setPlainLayout();
    }

    virtual services::Status setDimensions(const services::Collection<size_t>& dimensions) DAAL_C11_OVERRIDE
    {
        if(!dimensions.size())
        {
            return services::Status(services::ErrorNullParameterNotSupported);
        }

        _layout = data_management::TensorOffsetLayout(dimensions);
        return setPlainLayout();
    }

    services::Status assign(const DataType initValue)
    {
        size_t size = getSize();

        for(size_t i = 0; i < size; i++)
        {
            _plainPtr[i] = initValue;
        }

        _isDnnLayout = false;
        return services::Status();
    }

    services::Status freeDnnLayout();

    services::Status getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        data_management::ReadWriteMode rwflag, data_management::SubtensorDescriptor<double> &block,
                        const data_management::TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;
    services::Status getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        data_management::ReadWriteMode rwflag, data_management::SubtensorDescriptor<float> &block,
                        const data_management::TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;
    services::Status getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        data_management::ReadWriteMode rwflag, data_management::SubtensorDescriptor<int> &block,
                        const data_management::TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;

    services::Status getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        data_management::ReadWriteMode rwflag, data_management::SubtensorDescriptor<double>& subtensor ) DAAL_C11_OVERRIDE
    {
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, _layout );
    }

    services::Status getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        data_management::ReadWriteMode rwflag, data_management::SubtensorDescriptor<float>& subtensor ) DAAL_C11_OVERRIDE
    {
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, _layout );
    }

    services::Status getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        data_management::ReadWriteMode rwflag, data_management::SubtensorDescriptor<int>& subtensor ) DAAL_C11_OVERRIDE
    {
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, _layout );
    }

    services::Status releaseSubtensor(data_management::SubtensorDescriptor<double> &block) DAAL_C11_OVERRIDE;
    services::Status releaseSubtensor(data_management::SubtensorDescriptor<float>  &block) DAAL_C11_OVERRIDE;
    services::Status releaseSubtensor(data_management::SubtensorDescriptor<int>    &block) DAAL_C11_OVERRIDE;

    DAAL_DEPRECATED_VIRTUAL virtual data_management::TensorPtr getSampleTensor(size_t firstDimIndex) DAAL_C11_OVERRIDE
    {
        syncDnnToPlain();

        services::Collection<size_t> newDims = getDimensions();
        if(!_plainPtr || newDims.size() == 0 || newDims[0] <= firstDimIndex) { return data_management::TensorPtr(); }
        newDims[0] = 1;
        const size_t *_dimOffsets = &((_layout.getOffsets())[0]);
        services::Status st;
        return data_management::HomogenTensor<DataType>::create(newDims, _plainPtr + _dimOffsets[0]*firstDimIndex, &st);
    }

protected:
    virtual services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE;

    virtual services::Status freeDataMemoryImpl() DAAL_C11_OVERRIDE;

    services::Status serializeImpl  (data_management::InputDataArchive  *archive) DAAL_C11_OVERRIDE
    {
        serialImpl<data_management::InputDataArchive, false>( archive );

        return services::Status();
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {
        serialImpl<const data_management::OutputDataArchive, true>( archive );

        return services::Status();
    }

    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *archive )
    {
        Tensor::serialImpl<Archive, onDeserialize>( archive );

        archive->setObj( &_layout );

        if (!onDeserialize)
        {
            syncDnnToPlain();
        }
        else
        {
            freeDnnLayout();
            freePlainLayout();
            setPlainLayout();
        }

        bool isAllocated = (_memStatus != notAllocated);
        archive->set( isAllocated );

        if( onDeserialize )
        {
            freeDataMemory();

            if( isAllocated )
            {
                allocateDataMemory();
            }
        }

        _isDnnLayout = false;
        _isPlainLayout = false;

        if(_memStatus != notAllocated)
        {
            archive->set( _plainPtr, getSize() );

            _isPlainLayout = true;
        }
    }

private:
    template <typename T>
    services::Status getTSubtensor( size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, int rwFlag,
                        data_management::SubtensorDescriptor<T> &block, const data_management::TensorOffsetLayout& layout );
    template <typename T>
    services::Status releaseTSubtensor( data_management::SubtensorDescriptor<T> &block );

    services::Status freePlainLayout();
    services::Status setPlainLayout();
    services::Status syncPlainToDnn();

    MklTensor(services::Status &stat) :
        MklTensor() { }

    MklTensor(size_t nDim, const size_t *dimSizes, services::Status &stat) :
        MklTensor(nDim, dimSizes) { }

    MklTensor(size_t nDim, const size_t *dimSizes, AllocationFlag memoryAllocationFlag, services::Status &stat) :
        MklTensor(nDim, dimSizes, memoryAllocationFlag) { }

    MklTensor(const services::Collection<size_t> &dims, services::Status &stat) :
        MklTensor(dims) { }

    MklTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag, services::Status &stat) :
        MklTensor(dims, memoryAllocationFlag) { }

private:
    data_management::TensorOffsetLayout  _layout;

    DataType       *_dnnPtr;
    DnnLayoutPtr    _dnnLayout;
    bool            _isDnnLayout;

    DataType       *_plainPtr;
    DnnLayoutPtr    _plainLayout;
    bool            _isPlainLayout;
};


/**
 * Checks if tensor can be casted to %MklTensor
 */
template<typename algorithmFPType>
inline bool canCastToMklTensor(const data_management::Tensor *tensor)
{
    return dynamic_cast<const MklTensor<algorithmFPType> *>(tensor) != nullptr;
}

/**
 * Checks if tensor can be casted to %MklTensor
 */
template<typename algorithmFPType>
inline bool canCastToMklTensor(const data_management::Tensor &tensor)
{
    return canCastToMklTensor<algorithmFPType>(&tensor);
}

/**
 * Checks if tensor can be casted to %MklTensor
 */
template<typename algorithmFPType>
inline bool canCastToMklTensor(const data_management::TensorPtr &tensor)
{
    return canCastToMklTensor<algorithmFPType>(tensor.get());
}

/**
 * Checks if tensor can be casted to %MklTensor
 */
template<typename algorithmFPType>
inline bool isTensorInDnnLayout(const data_management::Tensor *tensor)
{
    using TensorType = MklTensor<algorithmFPType>;
    auto mklTensor = dynamic_cast<const TensorType *>(tensor);
    return mklTensor && const_cast<TensorType *>(mklTensor)->isDnnLayout();
}

/**
 * Checks if tensor can be casted to %MklTensor
 */
template<typename algorithmFPType>
inline bool isTensorInDnnLayout(const data_management::Tensor &tensor)
{
    return isTensorInDnnLayout<algorithmFPType>(&tensor);
}

/**
 * Checks if tensor can be casted to %MklTensor
 */
template<typename algorithmFPType>
inline bool isTensorInDnnLayout(const data_management::TensorPtr &tensor)
{
    return isTensorInDnnLayout<algorithmFPType>(tensor.get());
}

/**
 * If input tensor has DNN layout, function creates tensor with the same DNN layout,
 * otherwise it creates HomogenTensor<algorithmFPType>.
 */
template <typename algorithmFPType>
data_management::TensorPtr createTensorKeepingType(const data_management::Tensor *inputTensor,
                                                   services::Status &status)
{
    if (!inputTensor)
    {
        status.add(services::ErrorNullTensor);
        return data_management::TensorPtr();
    }

    if (canCastToMklTensor<algorithmFPType>(inputTensor))
    {
        return MklTensor<algorithmFPType>::create(inputTensor->getDimensions(), data_management::Tensor::doAllocate, &status);
    }
    else
    {
        using daal::data_management::HomogenTensor;
        return HomogenTensor<algorithmFPType>::create(inputTensor->getDimensions(), data_management::Tensor::doAllocate, &status);
    }
}


} // namespace internal
} // namespace daal

#endif
