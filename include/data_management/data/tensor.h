/* file: tensor.h */
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


#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "services/error_handling.h"
#include "services/daal_memory.h"
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
        notAllocate = 0, /*!< Memory will not be allocated by Tensor */
        doAllocate  = 1  /*!< Memory will be allocated by Tensor when needed */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__DATALAYOUT"></a>
     * \brief Enumeration to specify layout of Tensor data
     */
    enum DataLayout
    {
        defaultLayout = 0,      /*!< Default memory layout for the Tensor */
        unknownLayout = 0xffff, /*!< Unknown memory layout for the Tensor */
        rawLayout     = 0xffff, /*!< Raw memory layout for the Tensor */
    };

    virtual ~TensorIface()
    {}
    /**
     *  Sets the number of dimensions in the Tensor
     *
     *  \param[in] ndim     Number of dimensions
     *  \param[in] dimSizes Array with sizes for each dimension
     */
    virtual void setDimensions(size_t ndim, const size_t* dimSizes) = 0;

    /**
     *  Sets the number and size of dimensions in the Tensor
     *
     *  \param[in] dimensions Collection with sizes for each dimension
     */
    virtual void setDimensions(const services::Collection<size_t>& dimensions) = 0;

    /**
     *  Allocates memory for a data set
     */
    virtual void allocateDataMemory(daal::MemType type = daal::dram) = 0;

    /**
     *  Deallocates the memory allocated for a data set
     */
    virtual void freeDataMemory() = 0;

    /**
     * Checks the correctness of this tensor
     * \param[in] errors        Pointer to the collection of errors
     * \param[in] description   Additional information about error
     * \return Check status: True if the tensor satisfies the requirements, false otherwise.
     */
    virtual bool check(services::ErrorCollection *errors, const char *description) const = 0;

    /**
     *  Returns new tensor with first dimension limited to one point
     *  \param[in] firstDimIndex Index of the point in the first dimention
     */
    virtual services::SharedPtr<Tensor> getSampleTensor(size_t firstDimIndex) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__TENSORLAYOUTIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for representation of data layout in the tensor.
 *  This class declares the most general methods for data access.
 */
class DAAL_EXPORT TensorLayoutIface
{
public:
    DAAL_NEW_DELETE();
    virtual ~TensorLayoutIface() {}

    /**
     *  Sets the new order of existing dimension in the Tensor
     *
     *  \param[in] dimsOrder Collection with the new indices for each dimension
     */
    virtual void shuffleDimensions(const services::Collection<size_t>& dimsOrder) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__TENSORLAYOUT"></a>
 *  \brief Class for a data management component responsible for representation of data layout in the tensor.
 *  This class implements the most general methods for data layout.
 */
class DAAL_EXPORT TensorLayout : public TensorLayoutIface
{
public:
    /**
     *  Gets the layout type as DataLayout
     *
     *  \return The layout type as DataLayout
     */
    TensorIface::DataLayout getLayout() const
    {
        return _layout;
    }

    /**
     *  Checks if layout is equal to given
     *
     *  \param[in] layout The layout type to compare with
     *
     *  \return True or false
     */
    bool isLayout(TensorIface::DataLayout layout) const
    {
        return ((layout == _layout && _layout != TensorIface::unknownLayout) || layout == TensorIface::rawLayout);
    }

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
    TensorIface::DataLayout _layout;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__TENSOROFFSETLAYOUT"></a>
 *  \brief Class for a data management component responsible for representation of data layout in the HomogenTensor.
 */
class DAAL_EXPORT TensorOffsetLayout : public TensorLayout
{
public:
    TensorOffsetLayout(const TensorOffsetLayout& inLayout) : TensorLayout(inLayout.getDimensions()),
        _offsets(inLayout.getOffsets()), _isDefaultLayout(inLayout._isDefaultLayout)
    {
        _layout = inLayout.getLayout();
    }

    /**
     *  Constructor for TensorOffsetLayout with default layout
     *  \param[in]  dims  The size of dimensions in the Tensor layout
     */
    TensorOffsetLayout(const services::Collection<size_t>& dims) : TensorLayout(dims), _offsets(dims.size()), _isDefaultLayout(true)
    {
        if(_nDims==0) return;

        size_t lastIndex = _nDims-1;

        _offsets[lastIndex]=1;
        for(size_t i=1; i<_nDims; i++)
        {
            _offsets[lastIndex-i] = _offsets[lastIndex-i+1]*_dims[lastIndex-i+1];
        }

        _isDefaultLayout = true;
        _layout = TensorIface::defaultLayout;
    }

    /**
     *  Constructor for TensorOffsetLayout with layout defined with offsets between adjacent elements in each dimension
     *  \param[in]  dims     The size of dimensions in the Tensor layout
     *  \param[in]  offsets  The offsets between adjacent elements in each dimension
     */
    TensorOffsetLayout(const services::Collection<size_t>& dims, const services::Collection<size_t>& offsets) : TensorLayout(dims)
    {
        if(_nDims==0) return;
        if(dims.size()==offsets.size()) return;

        _offsets = offsets;

        checkLayout();
    }

    /**
     *  Gets the offsets between adjacent elements in each dimension
     *
     *  \return Collection with offsets for each dimension
     */
    const services::Collection<size_t>& getOffsets() const
    {
        return _offsets;
    }

    bool isDefaultLayout() const
    {
        return _isDefaultLayout;
    }

    virtual void shuffleDimensions(const services::Collection<size_t>& dimsOrder) DAAL_C11_OVERRIDE;

protected:
    services::Collection<size_t> _offsets;

    bool _isDefaultLayout;

private:
    void checkLayout();
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
    virtual void getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
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
    virtual void getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
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
    virtual void getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<int>& subtensor, const TensorOffsetLayout& layout ) = 0;

    /**
     *  Releases subtensor
     *
     *  \param[in] subtensor    The subtensor descriptor.
     */
    virtual void releaseSubtensor(SubtensorDescriptor<double>& subtensor) = 0;
    /**
     *  Releases subtensor
     *
     *  \param[in] subtensor    The subtensor descriptor.
     */
    virtual void releaseSubtensor(SubtensorDescriptor<float>& subtensor) = 0;
    /**
     *  Releases subtensor
     *
     *  \param[in] subtensor    The subtensor descriptor.
     */
    virtual void releaseSubtensor(SubtensorDescriptor<int>& subtensor) = 0;

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
    void getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<double>& subtensor )
    {
        getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, createDefaultSubtensorLayout() );
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
    void getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<float>& subtensor )
    {
        getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, createDefaultSubtensorLayout() );
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
    void getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<int>& subtensor )
    {
        getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, createDefaultSubtensorLayout() );
    }

    virtual TensorOffsetLayout createDefaultSubtensorLayout() const = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__TENSOR"></a>
 *  \brief Class for a data management component responsible for representation of data in the n-dimensions numeric format.
 *  This class implements the most general methods for data access.
 */
class DAAL_EXPORT Tensor : public SerializationIface, public TensorIface, public DenseTensorIface
{
public:
    /** \private */
    Tensor(TensorLayout *layoutPtr) : _errors(new services::KernelErrorCollection()), _layoutPtr(layoutPtr), _memStatus(notAllocated) {}

    /** \private */
    Tensor() : _errors(new services::KernelErrorCollection()), _layoutPtr(0), _memStatus(notAllocated) {}

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
     */
    services::SharedPtr<services::KernelErrorCollection> getErrors()
    {
        return _errors;
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
     * \param[in] errors        Pointer to the collection of errors
     * \param[in] description   Additional information about error
     * \return Check status: True if the tensor satisfies the requirements, false otherwise.
     */
    virtual bool check(services::ErrorCollection *errors, const char *description) const DAAL_C11_OVERRIDE
    {
        /* Check that the tensor is not empty */
        size_t nDims = getNumberOfDimensions();
        if (nDims == 0)
        {
            services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(services::ErrorIncorrectNumberOfDimensionsInTensor));
            error->addStringDetail(services::ArgumentName, description);
            errors->add(error);
            return false;
        }

        if (getSize() == 0)
        {
            services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(services::ErrorIncorrectSizeOfDimensionInTensor));
            error->addStringDetail(services::ArgumentName, description);
            errors->add(error);
            return false;
        }

        return true;
    }

    const TensorLayout* getLayoutPtr() const
    {
        return _layoutPtr;
    }

protected:
    MemoryStatus  _memStatus;
    services::SharedPtr<services::KernelErrorCollection> _errors;

private:
    TensorLayout *_layoutPtr;
};
typedef services::SharedPtr<Tensor> TensorPtr;
/** @} */
}

using interface1::Tensor;
using interface1::TensorPtr;
using interface1::TensorOffsetLayout;
using interface1::TensorLayout;

/**
 * Checks the correctness of this tensor
 * \param[in] tensor        Pointer to the tensor to check
 * \param[in] errors        Pointer to the collection of errors
 * \param[in] description   Additional information about error
 * \param[in] dims          Collection with required tensor dimension sizes
 * \return                  Check status:  True if the tensor satisfies the requirements, false otherwise.
 */
DAAL_EXPORT bool checkTensor(const Tensor *tensor, services::ErrorCollection *errors, const char *description,
                             const services::Collection<size_t> *dims = NULL);
}
} // namespace daal

#include "data_management/data/subtensor.h"

#endif
