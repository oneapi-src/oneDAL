/* file: subtensor.h */
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


#ifndef __SUBTENSOR_H__
#define __SUBTENSOR_H__

#include "services/error_handling.h"
#include "services/daal_memory.h"
#include "data_management/data/numeric_types.h"
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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SUBTENSORDESCRIPTOR"></a>
 *  \brief Class with descriptor of the subtensor retrieved from Tensor getSubTensor function
 */
template<typename DataType>
class DAAL_EXPORT SubtensorDescriptor
{
public:
    /** Constructor for an empty subtensor descriptor */
    SubtensorDescriptor();

    /** \private */
    ~SubtensorDescriptor();

    /**
     *   Gets a pointer to the buffer for the subtensor
     *  \return Pointer to the buffer for the subtensor
     */
    inline DataType* getPtr() const
    {
        if(_rawPtr)
        {
            return (DataType *)_rawPtr;
        }
        return _ptr.get();
    }

    /**
     *   Gets a pointer to the buffer for the subtensor
     *  \return Pointer to the buffer for the subtensor
     */
    inline services::SharedPtr<DataType> getSharedPtr() const
    {
        if(_rawPtr)
        {
            return services::SharedPtr<DataType>(services::reinterpretPointerCast<DataType, byte>(*_pPtr), (DataType *)_rawPtr);
        }
        return _ptr;
    }
    /**
     *  Returns the number of dimensions of the subtensor
     *  \return Number of dimensions of the subtensor
     */
    inline size_t getNumberOfDims() const { return _tensorNDims-_nFixedDims; }

    /**
     *  Returns the array with sizes of dimensions of the subtensor
     *  \return Array with sizes of the dimensions of the subtensor
     */
    inline size_t* getSubtensorDimSizes() const { return _dimNums+_nFixedDims; }

    /**
     *  Returns subtensor layout
     *  \return Layout
     */
    inline const TensorOffsetLayout *getLayout() const { return _layout; }

    /**
     *  Returns subtensor inplace flag
     *  \return Inplace flag
     */
    inline bool getInplaceFlag() const { return _inplaceFlag; }

    /**
     * Reset internal values and pointers to zero values
     */
    inline void reset()
    {
        _pPtr = NULL;
        _rawPtr = NULL;
    }

public:
    /**
     *  Sets data pointer to use for in-place calculation
     *  \param[in] ptr pointer
     *  \DAAL_DEPRECATED
     */
    inline void setPtr( DataType *ptr )
    {
        _ptr   = services::SharedPtr<DataType>(ptr, services::EmptyDeleter());
        _inplaceFlag = true;
    }

    inline void setPtr( services::SharedPtr<byte>* pPtr, byte * rawPtr )
    {
        _pPtr = pPtr;
        _rawPtr = rawPtr;
        _inplaceFlag = true;
    }

    /**
     *  Returns true if memory of (_subtensorSize) size is allocated successfully
     */
    inline bool resizeBuffer()
    {
        if ( _subtensorSize > _capacity )
        {
            freeBuffer();

            _buffer = services::SharedPtr<DataType>((DataType *)daal::services::daal_malloc(_subtensorSize * sizeof(DataType)), services::ServiceDeleter());

            if ( _buffer != 0 )
            {
                _capacity = _subtensorSize;
            }
            else
            {
                return false;
            }

        }

        _ptr = _buffer;
        _inplaceFlag = false;

        return true;
    }

    /**
     *  Sets subtensor parameters
     *  \param[in]  tensorNDims   The number of dimensions with fixed values
     *  \param[in]  tensorDimNums Values of the tensor dmensions
     *  \param[in]  nFixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums  Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx   Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum   Range for dimension values to get data from
     *  \param[in]  rwFlag        Flag specifying read/write access to the subtensor
     *  \return Subtensor size
     */
    size_t setDetails( size_t tensorNDims, const size_t *tensorDimNums,
        size_t nFixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, int rwFlag )
    {
        if (tensorDimNums == 0 || (nFixedDims > 0 && fixedDimNums == 0) || nFixedDims > tensorNDims)
        {
            return 0;
        }

        _rwFlag = rwFlag;

        if( _tensorNDims != tensorNDims )
        {
            if( _dimNums != _tensorNDimsBuffer )
            {
                daal::services::daal_free( _dimNums );
            }

            if( tensorNDims>10 )
            {
                _dimNums = (size_t*)daal::services::daal_malloc( tensorNDims * sizeof(size_t) );
            }
            else
            {
                _dimNums = _tensorNDimsBuffer;
            }

            if( !_dimNums )
            {
                _tensorNDims = 0;
                return 0;
            }

            _tensorNDims = tensorNDims;
        }

        _nFixedDims = nFixedDims;
        for( size_t i = 0; i < _nFixedDims; i++ )
        {
            _dimNums[i] = fixedDimNums[i];
        }

        _subtensorSize = 1;

        if( _nFixedDims != _tensorNDims )
        {
            _rangeDimIdx = rangeDimIdx;
            _dimNums[_nFixedDims] = rangeDimNum;
            _subtensorSize *= rangeDimNum;
        }

        for( size_t i = _nFixedDims+1; i < _tensorNDims; i++ )
        {
            _dimNums[i] = tensorDimNums[i];
            _subtensorSize *= tensorDimNums[i];
        }

        return _subtensorSize;
    }

    /**
     *  Saves subtensor offset layout
     *  \param[in]  layout  offset layout
     */
    inline bool saveOffsetLayout( const TensorOffsetLayout &layout )
    {
        if( !_layout )
        {
            _layout = const_cast<TensorOffsetLayout *>(&layout);
            _layoutOwnFlag = false;
        }
        return true;
    }

    /**
     *  Saves subtensor offset layout copy
     *  \param[in]  layout  offset layout
     */
    inline bool saveOffsetLayoutCopy( const TensorOffsetLayout &layout )
    {
        if( !_layout )
        {
            _layout = new TensorOffsetLayout(layout);
            if (!_layout)
            {
                return false;
            }
            _layoutOwnFlag = true;
        }
        return true;
    }

    /**
     *  Returns the full size of the subtensor in number of elements
     *  \return The full size of the subtensor in number of elements
     */
    inline size_t  getSize()         const { return _subtensorSize; }

    /**
     *  Gets the number of first dimension with fixed values
     *  \return The number of first dimension with fixed values
     */
    inline size_t  getFixedDims()    const { return _nFixedDims; }

    /**
     *  Gets values at which dimensions are fixed
     *  \return An array of values
     */
    inline size_t *getFixedDimNums() const { return _dimNums; }

    /**
     *  Gets values for the next dimension after fixed to get data from
     *  \return Value of the dimension
     */
    inline size_t  getRangeDimIdx()  const { return _rangeDimIdx; }

    /**
     *  Gets range for dimension values to get data from
     *  \return Range for dimension values to get data from
     */
    inline size_t  getRangeDimNum()  const
    {
        if( _nFixedDims != _tensorNDims )
        {
            return _dimNums[_nFixedDims];
        }
        return 1;
    }

    /**
     *  Returns a flag specifying read/write access to the subtensor
     *  \return Flag specifying read/write access to the subtensor
     */
    inline size_t  getRWFlag() const { return _rwFlag; }

protected:
    /**
     *  Frees the buffer
     */
    void freeBuffer()
    {
        _buffer = services::SharedPtr<DataType>();
        _capacity = 0;
    }

private:
    services::SharedPtr<DataType> _ptr;      /*<! Pointer to the buffer */
    services::SharedPtr<DataType> _buffer;   /*<! Pointer to the buffer */
    size_t    _capacity;                     /*<! Buffer size in bytes */

    size_t _tensorNDims;
    size_t _nFixedDims;
    size_t _rangeDimIdx;
    size_t *_dimNums;

    size_t _tensorNDimsBuffer[10];

    size_t _subtensorSize;

    int    _rwFlag;        /*<! Buffer size in bytes */
    TensorOffsetLayout *_layout;
    bool _layoutOwnFlag;
    bool _inplaceFlag;
    services::SharedPtr<byte> *_pPtr;
    byte *_rawPtr;
};
/** @} */

}

using interface1::SubtensorDescriptor;

}
} // namespace daal

#endif
