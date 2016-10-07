/* file: subtensor.h */
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
class SubtensorDescriptor
{
public:
    /** \private */
    SubtensorDescriptor() :
        _ptr(0), _buffer(0), _capacity(0),
        _tensorNDims(0), _nFixedDims(0), _rangeDimIdx(0), _dimNums(0),
        _subtensorSize(0), _rwFlag(0), _inplaceFlag(false), _layout(NULL) {}

    /** \private */
    ~SubtensorDescriptor()
    {
        freeBuffer();
        if( _dimNums )
        {
            daal::services::daal_free( _dimNums );
        }
        if( _layout )
        {
            delete _layout;
        }
    }

    /**
     *   Gets a pointer to the buffer for the subtensor
     *  \return Pointer to the subtensor
     */
    inline DataType* getPtr() const { return _ptr; }

    /**
     *  Returns the number of dimensions of the subtensor
     *  \return Number of columns
     */
    inline size_t getNumberOfDims() const { return _tensorNDims-_nFixedDims; }

    /**
     *  Returns the array with sizes of dimensions of the subtensor
     *  \return Number of rows
     */
    inline size_t* getSubtensorDimSizes() const { return _dimNums+_nFixedDims; }

    inline const TensorOffsetLayout* getLayout() const { return _layout; }

    inline bool getInplaceFlag() const { return _inplaceFlag; }

public:
    inline void setPtr( DataType* ptr )
    {
        _ptr = ptr;
        _inplaceFlag = true;
    }

    inline bool resizeBuffer()
    {
        if ( _subtensorSize > _capacity )
        {
            freeBuffer();

            _buffer = (DataType*)daal::services::daal_malloc( _subtensorSize*sizeof(DataType) );

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

    inline size_t setDetails( size_t tensorNDims, const size_t *tensorDimNums,
        size_t nFixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, int rwFlag )
    {
        _rwFlag = rwFlag;

        if( _tensorNDims != tensorNDims )
        {
            if( _tensorNDims != 0 )
            {
                daal::services::daal_free( _dimNums );
            }

            _dimNums = (size_t*)daal::services::daal_malloc( tensorNDims * sizeof(size_t) );
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

    inline void saveOffsetLayout( const TensorOffsetLayout & layout )
    {
        if( !_layout )
        {
            _layout = new TensorOffsetLayout(layout);
        }
    }

    inline size_t  getSize()         const { return _subtensorSize; }
    inline size_t  getFixedDims()    const { return _nFixedDims; }
    inline size_t* getFixedDimNums() const { return _dimNums; }
    inline size_t  getRangeDimIdx()  const { return _rangeDimIdx; }
    inline size_t  getRangeDimNum()  const
    {
        if( _nFixedDims != _tensorNDims )
        {
            return _dimNums[_nFixedDims];
        }
        return 1;
    }

    inline size_t  getRWFlag() const { return _rwFlag; }

protected:
    /**
     *  Frees the buffer
     */
    void freeBuffer()
    {
        if ( _capacity )
        {
            daal::services::daal_free( _buffer );
        }
        _buffer = 0;
        _capacity = 0;
    }

private:
    DataType *_ptr;      /*<! Pointer to the buffer */
    DataType *_buffer;   /*<! Pointer to the buffer */
    size_t    _capacity; /*<! Buffer size in bytes */

    size_t _tensorNDims;
    size_t _nFixedDims;
    size_t _rangeDimIdx;
    size_t *_dimNums;

    size_t _subtensorSize;

    int    _rwFlag;        /*<! Buffer size in bytes */
    TensorOffsetLayout *_layout;
    bool _inplaceFlag;
};
/** @} */

}

using interface1::SubtensorDescriptor;

}
} // namespace daal

#endif
