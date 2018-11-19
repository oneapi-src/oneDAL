/* file: tensor.h */
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


#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "services/error_handling.h"
#include "services/daal_memory.h"
#include "services/collection.h"
#include "data_management/data/data_archive.h"
#include "data_management/data/numeric_types.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @defgroup tensor Numeric Tensors
 * \brief Contains classes for a data management component responsible for representation of data in the n-dimensions numeric format.
 * @ingroup data_management
 * @{
 */
/**
 *  <a name="DAAL-CLASS-SUBTENSORDESCRIPTOR"></a>
 *  \brief %Base class that manages buffer memory for read/write operations required by tensors.
 */
template<typename DataType> class SubtensorDescriptor;

class Tensor;
typedef services::SharedPtr<Tensor> TensorPtr;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__TENSORIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for representation of data in the numeric format.
 *  This class declares the most general methods for data access.
 */
class TensorIface
{
public:
    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__MEMORYSTATUS"></a>
     * \brief Enumeration to specify the status of memory related to the Numeric Table
     */
    enum MemoryStatus
    {
        notAllocated        = 0, /*!< No memory allocated */
        userAllocated       = 1, /*!< Memory allocated on user side */
        internallyAllocated = 2  /*!< Memory allocated and managed by Tensor */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__ALLOCATIONFLAG"></a>
     * \brief Enumeration to specify whether the Numeric Table must allocate memory
     */
    enum AllocationFlag
    {
        doNotAllocate = 0, /*!< Memory will not be allocated by Tensor */
        notAllocate   = 0, /*!< Memory will not be allocated by Tensor \DAAL_DEPRECATED_USE{ \ref daal::data_management::interface1::TensorIface::doNotAllocate "doNotAllocate"}*/
        doAllocate    = 1  /*!< Memory will be allocated by Tensor when needed */
    };

    virtual ~TensorIface()
    {}
    /**
     *  Sets the number of dimensions in the Tensor
     *
     *  \param[in] ndim     Number of dimensions
     *  \param[in] dimSizes Array with sizes for each dimension
     */
    virtual services::Status setDimensions(size_t ndim, const size_t* dimSizes) = 0;

    /**
     *  Sets the number and size of dimensions in the Tensor
     *
     *  \param[in] dimensions Collection with sizes for each dimension
     */
    virtual services::Status setDimensions(const services::Collection<size_t>& dimensions) = 0;

    /**
     *  Allocates memory for a data set
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status allocateDataMemory(daal::MemType type = daal::dram) = 0;

    /**
     *  Deallocates the memory allocated for a data set
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status freeDataMemory() = 0;

    virtual services::Status resize(const services::Collection<size_t>& dimensions) = 0;

    /**
     * Checks the correctness of this tensor
     * \param[in] description   Additional information about error
     * \return Check status: True if the tensor satisfies the requirements, false otherwise.
     */
    virtual services::Status check(const char *description) const = 0;

    /**
     *  Returns new tensor with first dimension limited to one point
     *  \param[in] firstDimIndex Index of the point in the first dimention
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual TensorPtr getSampleTensor(size_t firstDimIndex) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__TENSORLAYOUTIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for representation of data layout in the tensor.
 *  This class declares the most general methods for data access.
 */
class DAAL_EXPORT TensorLayoutIface
{
public:
    virtual ~TensorLayoutIface() {}

    /**
     *  Sets the new order of existing dimension in the Tensor
     *
     *  \param[in] dimsOrder Collection with the new indices for each dimension
     */
    virtual services::Status shuffleDimensions(const services::Collection<size_t>& dimsOrder) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__TENSORLAYOUT"></a>
 *  \brief Class for a data management component responsible for representation of data layout in the tensor.
 *  This class implements the most general methods for data layout.
 */
class DAAL_EXPORT TensorLayout : public SerializationIface, public TensorLayoutIface
{
public:
    virtual ~TensorLayout() {}
    /**
     *  Gets the size of dimensions in the Tensor layout
     *
     *  \return Collection with sizes for each dimension
     */
    const services::Collection<size_t>& getDimensions() const
    {
        return _dims;
    }

protected:
    TensorLayout(const services::Collection<size_t>& dims) : _dims(dims), _nDims(dims.size()) {}

    size_t _nDims;
    services::Collection<size_t> _dims;

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl( Archive *arch )
    {
        arch->set(_dims);
        _nDims = _dims.size();

        return services::Status();
    }
};

typedef services::SharedPtr<TensorLayout> TensorLayoutPtr;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__TENSOROFFSETLAYOUT"></a>
 *  \brief Class for a data management component responsible for representation of data layout in the HomogenTensor.
 */
class DAAL_EXPORT TensorOffsetLayout : public TensorLayout
{
public:
    TensorOffsetLayout(const TensorOffsetLayout& inLayout) : TensorLayout(inLayout.getDimensions()),
        _offsets(inLayout.getOffsets()), _indices(inLayout.getIndices()), _isDefaultLayout(inLayout._isDefaultLayout),
        _isRawLayout(inLayout._isRawLayout)
    {}

    /**
     *  Constructor for TensorOffsetLayout with default layout
     *  \param[in]  dims  The size of dimensions in the Tensor layout
     */
    TensorOffsetLayout(const services::Collection<size_t>& dims) : TensorLayout(dims), _offsets(dims.size()), _indices(dims.size()),
                                                                   _isDefaultLayout(true), _isRawLayout(true)
    {
        if(_nDims==0) return;

        size_t lastIndex = _nDims-1;

        _offsets[lastIndex]=1;
        _indices[0] = 0;
        for(size_t i=1; i<_nDims; i++)
        {
            _offsets[lastIndex-i] = _offsets[lastIndex-i+1]*_dims[lastIndex-i+1];
            _indices[i] = i;
        }

        _isDefaultLayout = true;
        _isRawLayout = true;
    }

    /**
     *  Constructor for TensorOffsetLayout with layout defined with offsets between adjacent elements in each dimension
     *  \param[in]  dims     The size of dimensions in the Tensor layout
     *  \param[in]  offsets  The offsets between adjacent elements in each dimension
     *  \param[in]  indices  Collection with dimensions order
     */
    TensorOffsetLayout(const services::Collection<size_t>& dims, const services::Collection<size_t>& offsets,
                       const services::Collection<size_t>& indices) : TensorLayout(dims)
    {
        if(_nDims==0) return;
        if(dims.size()==offsets.size()) return;

        _offsets = offsets;
        _indices = indices;

        checkLayout();
    }

    virtual ~TensorOffsetLayout() {}

    /**
     *  Gets the offsets between adjacent elements in each dimension
     *
     *  \return Collection with offsets for each dimension
     */
    const services::Collection<size_t>& getOffsets() const
    {
        return _offsets;
    }

    /**
     *  Gets the dimensions order
     *
     *  \return Collection with dimensions order
     */
    const services::Collection<size_t>& getIndices() const
    {
        return _indices;
    }

    /**
     *  Checks if layout is equal to given
     *
     *  \param[in] layout The layout type to compare with
     *
     *  \return True or false
     */
    bool isLayout(const TensorOffsetLayout & layout) const
    {
        if( !(_nDims == layout.getDimensions().size()) ) return false;

        const services::Collection<size_t> & dims    = layout.getDimensions();
        const services::Collection<size_t> & offsets = layout.getOffsets();

        int dimsMatch = 0;
        int offsetsMatch = 0;
        for(size_t i = 0; i < _nDims; i++)
        {
            dimsMatch    += _dims[i]    == dims[i];
            offsetsMatch += _offsets[i] == offsets[i];
        }
        return (dimsMatch == _nDims) && (offsetsMatch == _nDims);
    }

    bool isDefaultLayout() const
    {
        return _isDefaultLayout;
    }

    bool isRawLayout() const
    {
        return _isRawLayout;
    }

    virtual services::Status shuffleDimensions(const services::Collection<size_t>& dimsOrder) DAAL_C11_OVERRIDE;

    services::Status sortOffsets();

    virtual int getSerializationTag() const DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_TENSOR_OFFSET_LAYOUT_ID;
    }

    DECLARE_SERIALIZABLE_IMPL();

protected:
    services::Collection<size_t> _offsets;
    services::Collection<size_t> _indices;

    bool _isDefaultLayout;
    bool _isRawLayout;

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl( Archive *arch )
    {
        TensorLayout::serialImpl<Archive,onDeserialize>(arch);

        arch->set(_offsets);
        arch->set(_indices);
        arch->set(_isDefaultLayout);
        arch->set(_isRawLayout);

        return services::Status();
    }

private:
    services::Status checkLayout();
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DENSETENSORIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for accessing data in the numeric format.
 *  This class declares specific methods to access Tensor data in a dense homogeneous form.
 */
class DenseTensorIface
{
public:
    virtual ~DenseTensorIface()
    {}
    /**
     *  Gets subtensor from the tensor
     *
     *  \param[in]  fixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx  Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum  Range for dimension values to get data from
     *  \param[in]  rwflag       Flag specifying read/write access to the subtensor
     *  \param[out] subtensor    The subtensor descriptor.
     *  \param[in]  layout       Layout of the requested subtensor
     */
    virtual services::Status getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<double>& subtensor, const TensorOffsetLayout& layout ) = 0;
    /**
     *  Gets subtensor from the tensor
     *
     *  \param[in]  fixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx  Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum  Range for dimension values to get data from
     *  \param[in]  rwflag       Flag specifying read/write access to the subtensor
     *  \param[out] subtensor    The subtensor descriptor.
     *  \param[in]  layout       Layout of the requested subtensor
     */
    virtual services::Status getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<float>& subtensor, const TensorOffsetLayout& layout ) = 0;
    /**
     *  Gets subtensor from the tensor
     *
     *  \param[in]  fixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx  Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum  Range for dimension values to get data from
     *  \param[in]  rwflag       Flag specifying read/write access to the subtensor
     *  \param[out] subtensor    The subtensor descriptor.
     *  \param[in]  layout       Layout of the requested subtensor
     */
    virtual services::Status getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<int>& subtensor, const TensorOffsetLayout& layout ) = 0;

    /**
     *  Releases subtensor
     *
     *  \param[in] subtensor    The subtensor descriptor.
     */
    virtual services::Status releaseSubtensor(SubtensorDescriptor<double>& subtensor) = 0;
    /**
     *  Releases subtensor
     *
     *  \param[in] subtensor    The subtensor descriptor.
     */
    virtual services::Status releaseSubtensor(SubtensorDescriptor<float>& subtensor) = 0;
    /**
     *  Releases subtensor
     *
     *  \param[in] subtensor    The subtensor descriptor.
     */
    virtual services::Status releaseSubtensor(SubtensorDescriptor<int>& subtensor) = 0;

    /**
     *  Gets subtensor from the tensor
     *
     *  \param[in]  fixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx  Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum  Range for dimension values to get data from
     *  \param[in]  rwflag       Flag specifying read/write access to the subtensor
     *  \param[out] subtensor    The subtensor descriptor.
     */
    virtual services::Status getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<double>& subtensor )
    {
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, createDefaultSubtensorLayout() );
    }

    /**
     *  Gets subtensor from the tensor
     *
     *  \param[in]  fixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx  Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum  Range for dimension values to get data from
     *  \param[in]  rwflag       Flag specifying read/write access to the subtensor
     *  \param[out] subtensor    The subtensor descriptor.
     */
    virtual services::Status getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<float>& subtensor )
    {
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, createDefaultSubtensorLayout() );
    }

    /**
     *  Gets subtensor from the tensor
     *
     *  \param[in]  fixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx  Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum  Range for dimension values to get data from
     *  \param[in]  rwflag       Flag specifying read/write access to the subtensor
     *  \param[out] subtensor    The subtensor descriptor.
     */
    virtual services::Status getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<int>& subtensor )
    {
        return getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, createDefaultSubtensorLayout() );
    }

    virtual TensorOffsetLayout createDefaultSubtensorLayout() const = 0;
    virtual TensorOffsetLayout createRawSubtensorLayout() const = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__TENSOR"></a>
 *  \brief Class for a data management component responsible for representation of data in the n-dimensions numeric format.
 *  This class implements the most general methods for data access.
 */
class DAAL_EXPORT Tensor : public SerializationIface, public TensorIface, public DenseTensorIface
{
public:
    DAAL_CAST_OPERATOR(Tensor)

    /**
     *  Constructor for a Tensor with predefined layout
     *  \param[in]  layoutPtr      Pointer to the Tensor Layout
     *  \DAAL_DEPRECATED
     */
    Tensor(TensorLayout *layoutPtr) : _layoutPtr(layoutPtr), _memStatus(notAllocated) {}

    /**
     *  Constructor for an empty Tenor
     */
    Tensor() : _layoutPtr(0), _memStatus(notAllocated) {}

    /** \private */
    virtual ~Tensor() {}

    /**
     *  Gets the status of the memory used by a data set connected with a Tensor
     */
    MemoryStatus getDataMemoryStatus() const { return _memStatus; }

    /**
     *  Gets the number of dimensions in the Tensor
     *
     *  \return Number of dimensions
     */
    size_t getNumberOfDimensions() const
    {
        return _layoutPtr->getDimensions().size();
    }

    /**
     *  Gets the size of the dimension in the Tensor
     *
     *  \param[in] dimIdx Index of dimension
     *
     *  \return Dimension size
     */
    size_t getDimensionSize(size_t dimIdx) const
    {
        if(getNumberOfDimensions() > dimIdx) return (_layoutPtr->getDimensions())[dimIdx];
        return 0;
    }

    /**
     *  Gets the size of dimensions in the Tensor
     *
     *  \return Collection with sizes for each dimension
     */
    const services::Collection<size_t>& getDimensions() const
    {
        return _layoutPtr->getDimensions();
    }

    /**
     *  Returns errors during the computation
     *  \return Errors during the computation
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::SharedPtr<services::KernelErrorCollection> getErrors()
    {
        return _status.getCollection()->getErrors();
    }

    /**
     *  Returns the full size of the tensor in number of elements
     *  \return The full size of the tensor in number of elements
     */
    size_t getSize() const;

    /**
     *  Returns the product of sizes of the range of dimensions
     *  \param[in] startingIdx The first dimension to include in the range
     *  \param[in] rangeSize   Number of dimensions to include in the range
     *  \return The product of sizes of the range of dimensions
     */
    size_t getSize(size_t startingIdx, size_t rangeSize) const;

    /**
     * Checks the correctness of this tensor
     * \param[in] description   Additional information about error
     * \return Check status: True if the tensor satisfies the requirements, false otherwise.
     */
    virtual services::Status check(const char *description) const DAAL_C11_OVERRIDE
    {
        if(_memStatus == notAllocated)
        {
            return services::Status(services::ErrorNullTensor);
        }
        /* Check that the tensor is not empty */
        size_t nDims = getNumberOfDimensions();
        if (nDims == 0)
        {
            return services::Status(services::ErrorIncorrectNumberOfDimensionsInTensor);
        }

        if (getSize() == 0)
        {
            return services::Status(services::ErrorIncorrectSizeOfDimensionInTensor);
        }

        return services::Status();
    }

    const TensorLayout* getLayoutPtr() const
    {
        return _layoutPtr;
    }

    DAAL_DEPRECATED_VIRTUAL services::Status allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        return allocateDataMemoryImpl(type);
    }

    DAAL_DEPRECATED_VIRTUAL services::Status freeDataMemory() DAAL_C11_OVERRIDE
    {
        return freeDataMemoryImpl();
    }

    virtual services::Status resize(const services::Collection<size_t>& dimensions) DAAL_C11_OVERRIDE
    {
        freeDataMemoryImpl();
        services::Status s = setDimensions(dimensions);
        if(!s)
            return s;
        s = allocateDataMemoryImpl();
        return s;
    }

protected:
    MemoryStatus  _memStatus;
    services::Status _status;

    Tensor(TensorLayout *layoutPtr, services::Status &st) : _layoutPtr(layoutPtr), _memStatus(notAllocated) {}

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl( Archive *arch )
    {
        if( onDeserialize )
        {
            _memStatus = notAllocated;
        }

        return services::Status();
    }

    virtual services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) = 0;

    virtual services::Status freeDataMemoryImpl() = 0;

private:
    TensorLayout *_layoutPtr;
};
/** @} */
}

using interface1::Tensor;
using interface1::TensorIface;
using interface1::TensorPtr;
using interface1::TensorOffsetLayout;
using interface1::TensorLayout;
using interface1::TensorLayoutPtr;

/**
 * Checks the correctness of this tensor
 * \param[in] tensor        Pointer to the tensor to check
 * \param[in] description   Additional information about error
 * \param[in] dims          Collection with required tensor dimension sizes
 * \return                  Check status:  True if the tensor satisfies the requirements, false otherwise.
 */
DAAL_EXPORT services::Status checkTensor(const Tensor *tensor, const char *description, const services::Collection<size_t> *dims = NULL);
}
} // namespace daal

#include "data_management/data/subtensor.h"

#endif
