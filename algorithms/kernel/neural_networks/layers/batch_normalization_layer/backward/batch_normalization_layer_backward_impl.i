/* file: batch_normalization_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of backward batch normalization layer
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_BACKWARD_IMPL_I__
#define __BATCH_NORMALIZATION_LAYER_BACKWARD_IMPL_I__

#include "service_math.h"
#include "threading.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace backward
{
namespace internal
{

/* pre-defined adjustable sizes */
#define _MIN_THREAD_SIZE_    16384
#define _MAX_BLOCK_KJ_SIZE_  1048576
#define _MIN_BLOCK_KJ_SIZE_  1024
#define _DEF_BLOCKS_MUL_     2

/* Common structure with arrays and variables */
template<typename algorithmFPType, Method method, CpuType cpu>
struct common_batchnorm_data_t
{
    /**************** Constructor **************************************************/
    common_batchnorm_data_t( Tensor *gradientTensor,
                             Tensor *weightsTensor,
                             Tensor *stDevTensor,
                             Tensor *inputGradientTensor,
                             Tensor *dataTensor,
                             Tensor *meanTensor,
                             Tensor *weightsDerTensor,
                             Tensor *biasesDerTensor,
                             const batch_normalization::Parameter *parameter )
    {
        _malloc_errors = 0;

        _gradientTensor           = gradientTensor;
        _weightsTensor            = weightsTensor;
        _stDevTensor              = stDevTensor;
        _inputGradientTensor      = inputGradientTensor;
        _dataTensor               = dataTensor;
        _meanTensor               = meanTensor;
        _weightsDerTensor         = weightsDerTensor;
        _biasesDerTensor          = biasesDerTensor;

        _propagate_gradient = parameter->propagateGradient;

        const services::Collection<size_t>& dims = inputGradientTensor->getDimensions();

        _dimension      = parameter->dimension;
        _dimensionSize  = dims[_dimension];
        _dimension0Size = dims[0];
        _nDims          = dims.size();

        _offsetBefore   = (_dimension == 0 ? 1 : inputGradientTensor->getSize(0, _dimension));
        _offsetAfter    = (_dimension == _nDims - 1 ? 1 : inputGradientTensor->getSize(_dimension + 1, _nDims - _dimension - 1));

        _weightsTensor        ->getSubtensor(0, 0, 0, _dimensionSize,  readOnly,  _weightsBlock);
        _stDevTensor          ->getSubtensor(0, 0, 0, _dimensionSize,  readOnly,  _stDevBlock);
        _inputGradientTensor  ->getSubtensor(0, 0, 0, _dimension0Size, readOnly,  _inputGradientBlock);
        _dataTensor           ->getSubtensor(0, 0, 0, _dimension0Size, readOnly,  _dataBlock);
        _meanTensor           ->getSubtensor(0, 0, 0, _dimensionSize,  readOnly,  _meanBlock);
        _weightsDerTensor     ->getSubtensor(0, 0, 0, _dimensionSize,  writeOnly, _weightsDerBlock);
        _biasesDerTensor      ->getSubtensor(0, 0, 0, _dimensionSize,  writeOnly, _biasesDerBlock);

        _weights       = _weightsBlock.getPtr();
        _stDev         = _stDevBlock.getPtr();
        _inputGradient = _inputGradientBlock.getPtr();
        _data          = _dataBlock.getPtr();
        _mean          = _meanBlock.getPtr();
        _weightsDer    = _weightsDerBlock.getPtr();
        _biasesDer     = _biasesDerBlock.getPtr();

        _invStDev  = (algorithmFPType *)daal_malloc(_dimensionSize * sizeof(algorithmFPType));

        if(  !(_weights) || !(_stDev) || !(_inputGradient) || !(_data) || !(_mean) || !(_weightsDer) || !(_biasesDer) || !(_invStDev) )
        {
            _malloc_errors++; return;
        }

        if(_propagate_gradient)
        {
            _gradientTensor->getSubtensor(0, 0, 0, _dimension0Size, writeOnly, _gradientBlock);
            _gradient      = _gradientBlock.getPtr();

            _invStDevByWeights     = (algorithmFPType *)daal_malloc(_dimensionSize * sizeof(algorithmFPType));
            _biasesDerMultiplier   = (algorithmFPType *)daal_malloc(_dimensionSize * sizeof(algorithmFPType));
            _weightsDerMultiplier  = (algorithmFPType *)daal_malloc(_dimensionSize * sizeof(algorithmFPType));

            if( !(_gradient) || !(_invStDevByWeights) || !(_biasesDerMultiplier) || !(_weightsDerMultiplier)  )
            {
                _malloc_errors++; return;
            }
        }

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for (size_t k = 0; k < _dimensionSize; k++)
        {
            _weightsDer[k]  = (algorithmFPType)0.0;
            _biasesDer[k]   = (algorithmFPType)0.0;
        }

        return;
    }
    /*******************************************************************************/

    /*********************** Destructor ********************************************/
    ~common_batchnorm_data_t()
    {

        if(_weights)_weightsTensor             ->releaseSubtensor(_weightsBlock);
        if(_stDev)_stDevTensor                 ->releaseSubtensor(_stDevBlock);
        if(_inputGradient)_inputGradientTensor ->releaseSubtensor(_inputGradientBlock);
        if(_data)_dataTensor                   ->releaseSubtensor(_dataBlock);
        if(_mean)_meanTensor                   ->releaseSubtensor(_meanBlock);
        if(_weightsDer)_weightsDerTensor       ->releaseSubtensor(_weightsDerBlock);
        if(_biasesDer)_biasesDerTensor         ->releaseSubtensor(_biasesDerBlock);

        if(_invStDev)daal_free(_invStDev);

        if(_propagate_gradient)
        {
            if(_gradient)_gradientTensor->releaseSubtensor(_gradientBlock);

            if(_invStDevByWeights)daal_free(_invStDevByWeights);
            if(_biasesDerMultiplier)daal_free(_biasesDerMultiplier);
            if(_weightsDerMultiplier)daal_free(_weightsDerMultiplier);
        }

        return;
    }
    /*******************************************************************************/

    /****************** Calculate weight and bias derivatives ***********************/
    void compute_weights_biases(size_t i_index, size_t kbeg, size_t kend)
    {
        for(size_t k = kbeg; k < kend; k++)
        {
            size_t idx_ik = ( i_index * _dimensionSize + k ) * _offsetAfter;

            algorithmFPType biasesDerSum  = 0.0;
            algorithmFPType weightsDerSum = 0.0;

            algorithmFPType m = _mean[k];
            algorithmFPType s = _invStDev[k];

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for ( size_t j = 0; j < _offsetAfter; j++ )
            {
                algorithmFPType g = _inputGradient[ idx_ik + j ];
                algorithmFPType d = _data[ idx_ik + j ];

                biasesDerSum   += g;
                weightsDerSum  += g * (d - m) * s;
            }

            _biasesDer[k]  += biasesDerSum;
            _weightsDer[k] += weightsDerSum;
        }

    return;
    }
    /*******************************************************************************/

    /****************** Calculate output gradients *********************************/
    void compute_gradients(size_t i_index, size_t kbeg, size_t kend)
    {
        for(size_t k = kbeg; k < kend; k++)
        {
            size_t idx_ik = ( i_index * _dimensionSize + k ) * _offsetAfter;

            algorithmFPType iw = _invStDevByWeights[k];
            algorithmFPType bm = _biasesDerMultiplier[k];
            algorithmFPType wm = _weightsDerMultiplier[k];
            algorithmFPType m  = _mean[k];

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for ( size_t j = 0; j < _offsetAfter; j++ )
            {
               algorithmFPType g = _inputGradient[ idx_ik + j ];
               algorithmFPType d = _data[ idx_ik + j ];

               _gradient[ idx_ik + j ] =  iw * ( g - bm - wm * ( d - m ) );
            }
        }

    return;
    }
    /*******************************************************************************/

    Tensor *_gradientTensor;
    Tensor *_weightsTensor;
    Tensor *_stDevTensor;
    Tensor *_inputGradientTensor;
    Tensor *_dataTensor;
    Tensor *_meanTensor;
    Tensor *_weightsDerTensor;
    Tensor *_biasesDerTensor;

    SubtensorDescriptor<algorithmFPType> _gradientBlock;
    SubtensorDescriptor<algorithmFPType> _weightsBlock;
    SubtensorDescriptor<algorithmFPType> _stDevBlock;
    SubtensorDescriptor<algorithmFPType> _inputGradientBlock;
    SubtensorDescriptor<algorithmFPType> _dataBlock;
    SubtensorDescriptor<algorithmFPType> _meanBlock;
    SubtensorDescriptor<algorithmFPType> _weightsDerBlock;
    SubtensorDescriptor<algorithmFPType> _biasesDerBlock;

    algorithmFPType *_gradient;
    algorithmFPType *_weights;
    algorithmFPType *_stDev;
    algorithmFPType *_inputGradient;
    algorithmFPType *_data;
    algorithmFPType *_mean;
    algorithmFPType *_weightsDer;
    algorithmFPType *_biasesDer;

    algorithmFPType *_invStDev;

    algorithmFPType *_invStDevByWeights;
    algorithmFPType *_biasesDerMultiplier;
    algorithmFPType *_weightsDerMultiplier;

    int _malloc_errors;
    int _propagate_gradient;

    size_t _dimension;
    size_t _dimension0Size;
    size_t _dimensionSize;
    size_t _nDims;
    size_t _offsetBefore;
    size_t _offsetAfter;
};

/* "Smart" block splitter */
/* computes block size (including last block) and number of blocks */
static inline void split_blocks(  int threadnum,
                                  size_t k_size,
                                  size_t j_size,
                                  int* pblocknum_k,
                                  int* pblocksize_k,
                                  int* pblocksize_last_k )
{
    /* Set initial number of blocks by */
    /* k = _DEF_BLOCKS_MUL_ * number_of_threads */
    /* _DEF_BLOCKS_MUL_ > 1 : number of blocks greater than threads */
    int _blocknum_k  = _DEF_BLOCKS_MUL_ * threadnum;
    int _blocksize_k;
    int _blocksize_last_k;

    /* If initial number of blocks greater than k-size, */
    /* set number of blocks = k-size */
    if(_blocknum_k > k_size)
    {
        _blocknum_k = k_size;
    }

    /* block size = k-size / number_of_blocks  */
    _blocksize_k = k_size / _blocknum_k;

    /* If block size is too big */
    if( (_blocksize_k * j_size) > _MAX_BLOCK_KJ_SIZE_ )
    {
        _blocksize_k = _MAX_BLOCK_KJ_SIZE_ / j_size;

        if(_blocksize_k < 1)
        {
            _blocksize_k = 1;
        }

        _blocknum_k  = k_size / _blocksize_k;
    }
    /* If block size is too small */
    else if( (_blocksize_k * j_size) < _MIN_BLOCK_KJ_SIZE_ )
    {
        _blocksize_k = _MIN_BLOCK_KJ_SIZE_ / j_size;
        _blocknum_k  = k_size / _blocksize_k;

        if(_blocknum_k < 1)
        {
            _blocknum_k = 1;
            _blocksize_k = k_size;
        }
    }

    /* Last block size is generally bigger */
    _blocksize_last_k = _blocksize_k + (k_size - _blocknum_k*_blocksize_k);

    *pblocknum_k       = _blocknum_k;
    *pblocksize_k      = _blocksize_k;
    *pblocksize_last_k = _blocksize_last_k;

    return;
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchNormalizationKernel<algorithmFPType, method, cpu>::compute(     Tensor *gradientTensor,
                                                                          Tensor *weightsTensor,
                                                                          Tensor *stDevTensor,
                                                                          Tensor *inputGradientTensor,
                                                                          Tensor *dataTensor,
                                                                          Tensor *meanTensor,
                                                                          Tensor *weightsDerTensor,
                                                                          Tensor *biasesDerTensor,
                                                                          const batch_normalization::Parameter *parameter )
{
    common_batchnorm_data_t<algorithmFPType,method,cpu> _cd( gradientTensor,
                                                             weightsTensor,
                                                             stDevTensor,
                                                             inputGradientTensor,
                                                             dataTensor,
                                                             meanTensor,
                                                             weightsDerTensor,
                                                             biasesDerTensor,
                                                             parameter );

    if(_cd._malloc_errors)
    {
        this->_errors->add(daal::services::ErrorMemoryAllocationFailed);
        DAAL_RETURN_STATUS()
    }

    size_t ij  = _cd._offsetBefore * _cd._offsetAfter;
    size_t ikj = ij  * _cd._dimensionSize;

    int threadnum        = services::Environment::getInstance()->getNumberOfThreads();
    int do_threading     = ( ikj > _MIN_THREAD_SIZE_ ) && ( threadnum > 1 );

    int blocknum_k       = 1;
    int blocksize_k      = _cd._dimensionSize;
    int blocksize_last_k = blocksize_k;
    int blocknum_ik      = _cd._offsetBefore;

    /* Split target dimension by blocks for threaded branches only */
    if( do_threading )
    {
        split_blocks( threadnum,
                      _cd._dimensionSize,
                      _cd._offsetAfter,
                      &blocknum_k,
                      &blocksize_k,
                      &blocksize_last_k );

        /* Each 'i' from _offsetBefore is treated as separate block in threading */
        blocknum_ik = _cd._offsetBefore * blocknum_k;
    }

   PRAGMA_NOVECTOR
    for (size_t k = 0; k < _cd._dimensionSize; k++)
    {
        _cd._invStDev[k] = algorithmFPType(1.0) / _cd._stDev[k];
    }

    /* Calculate weight and bias derivatives */
    if( do_threading )
    {
        daal::threader_for( blocknum_ik, blocknum_ik, [ & ](int block_ik)
        {
            int block_i = block_ik / blocknum_k;
            int block_k = block_ik % blocknum_k;

            size_t kbeg  = block_k * blocksize_k;
            size_t kend  = kbeg + (( block_k == blocknum_k-1)? blocksize_last_k : blocksize_k);

            _cd.compute_weights_biases( (size_t)block_i, kbeg, kend );
        } );
    }
    else
    {
        for (size_t i = 0; i < _cd._offsetBefore; i++)
        {
            _cd.compute_weights_biases( i, 0, _cd._dimensionSize );
        }
    }

    /* Calculate output gradients */
    if(_cd._propagate_gradient)
    {
        algorithmFPType invM  = 1.0 / (algorithmFPType)ij;
        algorithmFPType invM1 = 1.0 / (algorithmFPType)(ij - 1);

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for ( size_t k = 0; k < _cd._dimensionSize; k++ )
        {
            _cd._invStDevByWeights[k]    = _cd._weights[k] * _cd._invStDev[k];
            _cd._biasesDerMultiplier[k]  = invM * _cd._biasesDer[k];
            _cd._weightsDerMultiplier[k] = invM1 * _cd._invStDev[k] * _cd._weightsDer[k];
        }

        if( do_threading ) /* threaded branch */
        {
            daal::threader_for( blocknum_ik, blocknum_ik, [ & ]( int block_ik )
            {
                int block_i = block_ik / blocknum_k;
                int block_k = block_ik % blocknum_k;

                size_t kbeg  = block_k * blocksize_k;
                size_t kend  = kbeg + (( block_k == (blocknum_k-1) )? blocksize_last_k : blocksize_k);

                _cd.compute_gradients( (size_t)block_i, kbeg, kend);
            } ); /* daal::threader_for */
        }
        else /* sequential branch */
        {
            for (size_t i = 0; i < _cd._offsetBefore; i++)
            {
                _cd.compute_gradients( i, 0, _cd._dimensionSize );
            }
        }
    }

    DAAL_RETURN_STATUS()
}

} // namespace internal
} // namespace backward
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
