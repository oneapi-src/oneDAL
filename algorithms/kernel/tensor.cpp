/** file tensor.cpp */
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

#include "services/daal_defines.h"
#include "services/collection.h"
#include "data_utils.h"
#include "tensor.h"
#include "homogen_tensor.h"

/**
 * Checks the correctness of this tensor
 * \param[in] tensor        Pointer to the tensor to check
 * \param[in] errors        Pointer to the collection of errors
 * \param[in] description   Additional information about error
 * \param[in] dims          Collection with required tensor dimension sizes
 * \return                  Check status:  True if the tensor satisfies the requirements, false otherwise.
 */
bool daal::data_management::checkTensor(const Tensor *tensor, services::ErrorCollection *errors,
                                        const char *description, const services::Collection<size_t> *dims)
{
    using namespace daal::services;

    if (tensor == 0)
    {
        SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorNullTensor));
        error->addStringDetail(ArgumentName, description);
        errors->add(error);
        return false;
    }

    if (dims)
    {
        /* Here if collection of the required dimension sizes is provided */
        if (tensor->getNumberOfDimensions() != dims->size())
        {
            SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorIncorrectNumberOfDimensionsInTensor));
            error->addStringDetail(ArgumentName, description);
            errors->add(error);
            return false;
        }

        for (size_t d = 0; d < dims->size(); d++)
        {
            if (tensor->getDimensionSize(d) != (*dims)[d])
            {
                SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorIncorrectSizeOfDimensionInTensor));
                error->addStringDetail(ArgumentName, description);
                error->addIntDetail(Dimension, (int)d);
                errors->add(error);
                return false;
            }
        }
    }

    return tensor->check(errors, description);
}

/**
 *  Returns the full size of the tensor in number of elements
 *  \return The full size of the tensor in number of elements
 */
size_t daal::data_management::Tensor::getSize() const
{
        size_t nDim = getNumberOfDimensions();
        if( nDim==0 ) return 0;

        size_t size = 1;

        for(size_t i=0; i<nDim; i++)
        {
            size *= (_layoutPtr->getDimensions())[i];
        }

        return size;
}

/**
 *  Returns the product of sizes of the range of dimensions
 *  \param[in] startingIdx The first dimension to include in the range
 *  \param[in] rangeSize   Number of dimensions to include in the range
 *  \return The product of sizes of the range of dimensions
 */
size_t daal::data_management::Tensor::getSize(size_t startingIdx, size_t rangeSize) const
{
        size_t nDim = getNumberOfDimensions();
        if( nDim==0 || rangeSize==0 || startingIdx>=nDim || startingIdx+rangeSize > nDim ) return 0;

        size_t size = 1;

        for(size_t i=0; i<rangeSize; i++)
        {
            size *= (_layoutPtr->getDimensions())[startingIdx+i];
        }

        return size;
}

namespace daal
{
namespace data_management
{
namespace interface1
{

void TensorOffsetLayout::shuffleDimensions(const services::Collection<size_t>& dimsOrder)
{
    services::Collection<size_t> newDims   (dimsOrder.size());
    services::Collection<size_t> newOffsets(dimsOrder.size());

    for(size_t i=0;i<_nDims;i++)
    {
        newDims   [ dimsOrder[i] ] = _dims   [i];
        newOffsets[ dimsOrder[i] ] = _offsets[i];
    }
    _dims    = newDims;
    _offsets = newOffsets;

    checkLayout();
}

void TensorOffsetLayout::checkLayout()
{
    size_t lastIndex = _nDims-1;

    int defaultLayoutMatch = (_offsets[lastIndex] == 1);
    for(size_t i=1; i<_nDims; i++)
    {
        defaultLayoutMatch += (_offsets[lastIndex-i] == _offsets[lastIndex-i+1]*_dims[lastIndex-i+1]);
    }

    _isDefaultLayout = ( defaultLayoutMatch==_nDims );

    if( defaultLayoutMatch==_nDims )
    {
        _layout = TensorIface::defaultLayout;
    }
    else
    {
        _layout = TensorIface::unknownLayout;
    }
}


template <typename DataType>
template <typename T>
void HomogenTensor<DataType>::getTSubtensor( size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                                             int rwFlag, SubtensorDescriptor<T> &block, const TensorOffsetLayout& layout )
{
    size_t  nDim     = layout.getDimensions().size();
    const size_t *dimSizes    = &((layout.getDimensions())[0]);
    const size_t *_dimOffsets = &((layout.getOffsets())[0]);

    size_t blockSize = block.setDetails( nDim, dimSizes, fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwFlag );
    block.saveOffsetLayout(layout);

    size_t shift = 0;
    for( size_t i = 0; i < fixedDims; i++ )
    {
        shift += fixedDimNums[i] * _dimOffsets[i];
    }
    if( fixedDims != nDim )
    {
        shift += rangeDimIdx * _dimOffsets[fixedDims];
    }

    if( layout.isDefaultLayout() )
    {
        if( IsSameType<T, DataType>::value )
        {
            block.setPtr( (T *)(_ptr + shift) );
        }
        else
        {
            if( !block.resizeBuffer() )
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }

            if( rwFlag & (int)readOnly )
            {
                data_feature_utils::vectorUpCast[data_feature_utils::getIndexNumType<DataType>()][data_feature_utils::getInternalNumType<T>()]
                ( blockSize, _ptr + shift, block.getPtr() );
            }
        }
    }
    else
    {
        if( !block.resizeBuffer() )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        if( rwFlag & (int)readOnly )
        {
            size_t leftDims = nDim - fixedDims;

            size_t* bIdxs = new size_t[leftDims];
            size_t* bDims = new size_t[leftDims];

            bIdxs[0] = 0;
            bDims[0] = rangeDimNum;
            for( size_t i=1; i<leftDims; i++ )
            {
                bIdxs[i] = 0;
                bDims[i] = dimSizes[fixedDims+i];
            }

            for( size_t b=0; b<blockSize; b++ )
            {
                size_t rShift = 0;
                for( size_t i=0; i<leftDims; i++ )
                {
                    rShift += bIdxs[ i ]*_dimOffsets[fixedDims+i];
                }

                *(block.getPtr() + b) = *(_ptr + shift + rShift);

                for( size_t i=0; i<leftDims; i++ )
                {
                    bIdxs[ leftDims-1-i ]++;
                    if( bIdxs[ leftDims-1-i ] < bDims[ leftDims-1-i ] ) break;
                    bIdxs[ leftDims-1-i ] = 0;
                }
            }

            delete[] bDims;
            delete[] bIdxs;
        }
    }
}

template <typename DataType>
template <typename T>
void HomogenTensor<DataType>::releaseTSubtensor( SubtensorDescriptor<T> &block )
{
    if( (block.getRWFlag() & (int)writeOnly) && !block.getInplaceFlag() )
    {
        if( block.getLayout()->isDefaultLayout() )
        {
            if( !IsSameType<T, DataType>::value )
            {
                const size_t *_dimOffsets = &((block.getLayout()->getOffsets())[0]);

                size_t nDim = getNumberOfDimensions();

                size_t blockSize = block.getSize();

                size_t fixedDims     = block.getFixedDims();
                size_t *fixedDimNums = block.getFixedDimNums();
                size_t rangeDimIdx   = block.getRangeDimIdx();

                size_t shift = 0;
                for( size_t i = 0; i < fixedDims; i++ )
                {
                    shift += fixedDimNums[i] * _dimOffsets[i];
                }
                if( fixedDims != nDim )
                {
                    shift += rangeDimIdx * _dimOffsets[fixedDims];
                }

                data_feature_utils::vectorDownCast[data_feature_utils::getIndexNumType<DataType>()][data_feature_utils::getInternalNumType<T>()]
                ( blockSize, block.getPtr(), _ptr + shift );
            }
        }
        else
        {
            size_t nDim = getNumberOfDimensions();

            const size_t *dimSizes    = &((block.getLayout()->getDimensions())[0]);
            const size_t *_dimOffsets = &((block.getLayout()->getOffsets())[0]);

            size_t blockSize = block.getSize();

            size_t fixedDims     = block.getFixedDims();
            size_t *fixedDimNums = block.getFixedDimNums();
            size_t rangeDimIdx   = block.getRangeDimIdx();
            size_t rangeDimNum   = block.getRangeDimNum();

            size_t shift = 0;
            for( size_t i = 0; i < fixedDims; i++ )
            {
                shift += fixedDimNums[i] * _dimOffsets[i];
            }
            if( fixedDims != nDim )
            {
                shift += rangeDimIdx * _dimOffsets[fixedDims];
            }

            size_t leftDims = nDim - fixedDims;

            size_t* bIdxs = new size_t[leftDims];
            size_t* bDims = new size_t[leftDims];

            bIdxs[0] = 0;
            bDims[0] = rangeDimNum;
            for( size_t i=1; i<leftDims; i++ )
            {
                bIdxs[i] = 0;
                bDims[i] = dimSizes[fixedDims+i];
            }

            for( size_t b=0; b<blockSize; b++ )
            {
                size_t rShift = 0;
                for( size_t i=0; i<leftDims; i++ )
                {
                    rShift += bIdxs[ i ]*_dimOffsets[fixedDims+i];
                }

                *(_ptr + shift + rShift) = *(block.getPtr() + b);

                for( size_t i=0; i<leftDims; i++ )
                {
                    bIdxs[ leftDims-1-i ]++;
                    if( bIdxs[ leftDims-1-i ] < bDims[ leftDims-1-i ] ) break;
                    bIdxs[ leftDims-1-i ] = 0;
                }
            }

            delete[] bDims;
            delete[] bIdxs;
        }
    }
}

#define DAAL_IMPL_GETSUBTENSOR(T1,T2)                                                                                        \
template<>                                                                                                                   \
void HomogenTensor<T1>::getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, \
                  ReadWriteMode rwflag, SubtensorDescriptor<T2> &block, const TensorOffsetLayout& layout )                   \
{                                                                                                                            \
    return getTSubtensor<T2>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, block, layout);                      \
}

#define DAAL_IMPL_RELEASESUBTENSOR(T1,T2)                                 \
template<>                                                                \
void HomogenTensor<T1>::releaseSubtensor(SubtensorDescriptor<T2> &block)  \
{                                                                         \
    releaseTSubtensor<T2>(block);                                         \
}

#define DAAL_IMPL_HOMOGENTENSORCONSTRUCTOR(T)                                                                        \
template<>                                                                                                           \
HomogenTensor<T>::HomogenTensor(const services::Collection<size_t> &dims, T *data) : Tensor(&_layout), _layout(dims) \
{                                                                                                                    \
    _ptr = data;                                                                                                     \
    _allocatedSize = 0;                                                                                              \
    if( data )                                                                                                       \
    {                                                                                                                \
        _allocatedSize = getSize();                                                                                  \
    }                                                                                                                \
    size_t nDim = dims.size();                                                                                       \
    if(nDim == 0)                                                                                                    \
    {                                                                                                                \
        this->_errors->add(services::ErrorNullParameterNotSupported);                                                \
        return;                                                                                                      \
    }                                                                                                                \
}

#define DAAL_INSTANTIATE(T1,T2)                                                                                                                         \
template void HomogenTensor<T1>::getTSubtensor( size_t, const size_t *, size_t, size_t, int, SubtensorDescriptor<T2> &, const TensorOffsetLayout& );    \
template void HomogenTensor<T1>::releaseTSubtensor( SubtensorDescriptor<T2> & );                                                                        \
DAAL_IMPL_GETSUBTENSOR(T1,T2)                                                                                                                           \
DAAL_IMPL_RELEASESUBTENSOR(T1,T2)

#define DAAL_INSTANTIATE_THREE(T1)     \
DAAL_INSTANTIATE(T1, double)           \
DAAL_INSTANTIATE(T1, float )           \
DAAL_INSTANTIATE(T1, int   )           \
DAAL_IMPL_HOMOGENTENSORCONSTRUCTOR(T1)

DAAL_INSTANTIATE_THREE(float         )
DAAL_INSTANTIATE_THREE(double        )
DAAL_INSTANTIATE_THREE(int           )
DAAL_INSTANTIATE_THREE(unsigned int  )
DAAL_INSTANTIATE_THREE(DAAL_INT64    )
DAAL_INSTANTIATE_THREE(DAAL_UINT64   )
DAAL_INSTANTIATE_THREE(char          )
DAAL_INSTANTIATE_THREE(unsigned char )
DAAL_INSTANTIATE_THREE(short         )
DAAL_INSTANTIATE_THREE(unsigned short)
DAAL_INSTANTIATE_THREE(unsigned long )

}
}
}
