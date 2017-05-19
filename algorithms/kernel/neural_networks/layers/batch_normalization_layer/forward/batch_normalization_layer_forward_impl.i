/* file: batch_normalization_layer_forward_impl.i */
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
//  Implementation of forward batch normalization layer
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_IMPL_I__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_IMPL_I__

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
namespace forward
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
    common_batchnorm_data_t( Tensor *inputTensor,
                             Tensor *inPopMeanTensor,
                             Tensor *inPopVarianceTensor,
                             Tensor *weightsTensor,
                             Tensor *biasesTensor,
                             Tensor *popMeanTensor,
                             Tensor *popVarianceTensor,
                             Tensor *resultTensor,
                             Tensor *meanTensor,
                             Tensor *stDevTensor,
                             const batch_normalization::Parameter *parameter )
    {
        _malloc_errors = 0;

        _inputTensor         = inputTensor;
        _inPopMeanTensor     = inPopMeanTensor;
        _inPopVarianceTensor = inPopVarianceTensor;
        _weightsTensor       = weightsTensor;
        _biasesTensor        = biasesTensor;
        _popMeanTensor       = popMeanTensor;
        _popVarianceTensor   = popVarianceTensor;
        _resultTensor        = resultTensor;
        _meanTensor          = meanTensor;
        _stDevTensor         = stDevTensor;

        _prediction_stage = parameter->predictionStage;
        _epsilon          = (algorithmFPType)( parameter->epsilon );

        const services::Collection<size_t>& dims = inputTensor->getDimensions();

        _dimension      = parameter->dimension;
        _dimensionSize  = dims[_dimension];
        _dimension0Size = dims[0];
        _offsetBefore = 1;
        _offsetAfter  = 1;

        for (size_t i = 0; i < _dimension; i++)
        {
            _offsetBefore *= dims[i];
        }
        for (size_t i = _dimension + 1; i < dims.size(); i++)
        {
            _offsetAfter *= dims[i];
        }

        size_t ij  = _offsetBefore * _offsetAfter;
        _invM      = 1.0 / (algorithmFPType)ij;
        _invM1     = 1.0 / (algorithmFPType)(ij - 1);


        if( _prediction_stage == false )
        {
            _inPopMeanTensor     ->getSubtensor(0, 0, 0, _dimensionSize, readOnly,  _inPopMeanBlock);
            _inPopVarianceTensor ->getSubtensor(0, 0, 0, _dimensionSize, readOnly,  _inPopVarianceBlock);
            _popMeanTensor       ->getSubtensor(0, 0, 0, _dimensionSize, writeOnly, _popMeanBlock);
            _popVarianceTensor   ->getSubtensor(0, 0, 0, _dimensionSize, writeOnly, _popVarianceBlock);

            _inPopMean     = _inPopMeanBlock.getPtr();
            _inPopVariance = _inPopVarianceBlock.getPtr();
            _popMean       = _popMeanBlock.getPtr();
            _popVariance   = _popVarianceBlock.getPtr();

            if( !(_inPopMean) || !(_inPopVariance) || !(_popMean) || !(_popVariance))
            {
                _malloc_errors++; return;
            }
        }

        _inputTensor  ->getSubtensor(0, 0, 0, _dimension0Size, readOnly, _inputBlock);
        _meanTensor   ->getSubtensor(0, 0, 0, _dimensionSize, writeOnly, _meanBlock);
        _stDevTensor  ->getSubtensor(0, 0, 0, _dimensionSize, writeOnly, _stDevBlock);
        _weightsTensor->getSubtensor(0, 0, 0, _dimensionSize, readOnly, _weightsBlock);
        _biasesTensor ->getSubtensor(0, 0, 0, _dimensionSize, readOnly, _biasesBlock);
        _resultTensor ->getSubtensor(0, 0, 0, (_resultTensor->getDimensions())[0], writeOnly, _resultBlock);

        _input   = _inputBlock.getPtr();
        _mean    = _meanBlock.getPtr();
        _stdev   = _stDevBlock.getPtr();
        _weights = _weightsBlock.getPtr();
        _biases  = _biasesBlock.getPtr();
        _result  = _resultBlock.getPtr();

        _invstdw  = (algorithmFPType *)daal_malloc(_dimensionSize * sizeof(algorithmFPType));
        _variance = (algorithmFPType *)daal_malloc(_dimensionSize * sizeof(algorithmFPType));

        if( !(_input) || !(_mean) || !(_stdev) || !(_weights) || !(_biases) || !(_result) || !(_invstdw) || !(_variance) )
        {
            _malloc_errors++; return;
        }

        for (size_t i = 0; i < _dimensionSize; i++)
        {
            _mean[i]  = (algorithmFPType)0.0;
            _stdev[i] = (algorithmFPType)0.0;
        }

    return;
    }
    /*******************************************************************************/

    /************ Destructor *******************************************************/
    ~common_batchnorm_data_t()
    {
        if( _prediction_stage == false )
        {
            if(_inPopMean)_inPopMeanTensor        ->releaseSubtensor(_inPopMeanBlock);
            if(_inPopVariance)_inPopVarianceTensor->releaseSubtensor(_inPopVarianceBlock);
            if(_popMean)_popMeanTensor            ->releaseSubtensor(_popMeanBlock);
            if(_popVariance)_popVarianceTensor    ->releaseSubtensor(_popVarianceBlock);
        }

        if(_input)_inputTensor    ->releaseSubtensor(_inputBlock);
        if(_mean)_meanTensor      ->releaseSubtensor(_meanBlock);
        if(_stdev)_stDevTensor    ->releaseSubtensor(_stDevBlock);
        if(_weights)_weightsTensor->releaseSubtensor(_weightsBlock);
        if(_biases)_biasesTensor  ->releaseSubtensor(_biasesBlock);
        if(_result)_resultTensor  ->releaseSubtensor(_resultBlock);

        if(_invstdw)daal_free(_invstdw);
        if(_variance)daal_free(_variance);

    return;
    }
    /*******************************************************************************/

    /****************** Calculate sums of input tensor *****************************/
    void compute_sums( size_t i_index, size_t kbeg, size_t kend )
    {
        for(size_t k = kbeg; k < kend; k++)
        {
            size_t idx_ik = ( i_index * _dimensionSize + k ) * _offsetAfter;

            algorithmFPType sum   = 0.0;
            algorithmFPType sumSq = 0.0;

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for ( size_t j = 0; j < _offsetAfter; j++ )
            {
                algorithmFPType arg = _input[ idx_ik + j ];
                sum   += arg;
                sumSq += arg * arg;
            }

            _mean[k]  += sum;
            _stdev[k] += sumSq;
        }

    return;
    }
    /*******************************************************************************/

    /****************** Calculate means, stddevs... ********************************/
    void compute_means_stdevs( size_t kbeg, size_t kend )
    {

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for( size_t k = kbeg; k < kend; k++ )
        {
            _variance[k] = _invM1 * (_stdev[k] - _mean[k] * _mean[k] * _invM);
            _mean[k]    *= _invM;
            _stdev[k]    = _variance[k] + _epsilon;
            _invstdw[k]  = _weights[k] / _stdev[k];
        }

        daal::internal::Math<algorithmFPType,cpu>::vSqrt( (kend-kbeg), &(_stdev[kbeg]), &(_stdev[kbeg]));

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for( size_t k = kbeg; k < kend; k++ )
        {
            _invstdw[k]  = _weights[k] / _stdev[k];
        }

    return;
    }
    /*******************************************************************************/

    /***************** Final result gradient compute *******************************/
    void compute_gradients( size_t i_index, size_t kbeg, size_t kend )
    {
        for(size_t k = kbeg; k < kend; k++)
        {
            size_t idx_ik = ( i_index * _dimensionSize + k ) * _offsetAfter;

            algorithmFPType w = _invstdw[k];
            algorithmFPType m = _mean[k];
            algorithmFPType b = _biases[k];

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for ( size_t j = 0; j < _offsetAfter; j++ )
            {
                _result[ idx_ik + j ] = w * ( _input[ idx_ik + j ] - m ) + b;
            }
        }

    return;
    }
    /*******************************************************************************/

    Tensor *_inputTensor;
    Tensor *_inPopMeanTensor;
    Tensor *_inPopVarianceTensor;
    Tensor *_weightsTensor;
    Tensor *_biasesTensor;
    Tensor *_popMeanTensor;
    Tensor *_popVarianceTensor;
    Tensor *_resultTensor;
    Tensor *_meanTensor;
    Tensor *_stDevTensor;

    SubtensorDescriptor<algorithmFPType> _inputBlock;
    SubtensorDescriptor<algorithmFPType> _meanBlock;
    SubtensorDescriptor<algorithmFPType> _stDevBlock;
    SubtensorDescriptor<algorithmFPType> _inPopMeanBlock;
    SubtensorDescriptor<algorithmFPType> _inPopVarianceBlock;
    SubtensorDescriptor<algorithmFPType> _popMeanBlock;
    SubtensorDescriptor<algorithmFPType> _popVarianceBlock;
    SubtensorDescriptor<algorithmFPType> _weightsBlock;
    SubtensorDescriptor<algorithmFPType> _biasesBlock;
    SubtensorDescriptor<algorithmFPType> _resultBlock;

    algorithmFPType *_input;
    algorithmFPType *_mean;
    algorithmFPType *_stdev;
    algorithmFPType *_inPopMean;
    algorithmFPType *_inPopVariance;
    algorithmFPType *_popMean;
    algorithmFPType *_popVariance;
    algorithmFPType *_weights;
    algorithmFPType *_biases;
    algorithmFPType *_result;

    algorithmFPType *_invstdw;
    algorithmFPType *_variance;

    algorithmFPType _invM;
    algorithmFPType _invM1;
    algorithmFPType _epsilon;

    int _malloc_errors;
    int _prediction_stage;

    size_t _dimension;
    size_t _dimension0Size;
    size_t _dimensionSize;
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
services::Status BatchNormalizationKernel<algorithmFPType, method, cpu>::compute(     Tensor *inputTensor,
                                                                          Tensor *inPopMeanTensor,
                                                                          Tensor *inPopVarianceTensor,
                                                                          Tensor *weightsTensor,
                                                                          Tensor *biasesTensor,
                                                                          Tensor *popMeanTensor,
                                                                          Tensor *popVarianceTensor,
                                                                          Tensor *resultTensor,
                                                                          Tensor *meanTensor,
                                                                          Tensor *stDevTensor,
                                                                          const batch_normalization::Parameter *parameter )
{
    common_batchnorm_data_t<algorithmFPType,method,cpu> _cd( inputTensor,
                                                             inPopMeanTensor,
                                                             inPopVarianceTensor,
                                                             weightsTensor,
                                                             biasesTensor,
                                                             popMeanTensor,
                                                             popVarianceTensor,
                                                             resultTensor,
                                                             meanTensor,
                                                             stDevTensor,
                                                             parameter );

    if(_cd._malloc_errors)
    {
        this->_errors->add(daal::services::ErrorMemoryAllocationFailed);
        DAAL_RETURN_STATUS()
    }

    int threadnum        = services::Environment::getInstance()->getNumberOfThreads();
    size_t ikj           = _cd._offsetBefore * _cd._offsetAfter  * _cd._dimensionSize;
    int do_threading     = ( ikj > _MIN_THREAD_SIZE_ ) && ( threadnum > 1 );

    int blocknum_k       = 1;
    int blocksize_k      = _cd._dimensionSize;
    int blocksize_last_k = blocksize_k;
    int blocknum_ik      = _cd._offsetBefore;

    if( do_threading )
    {
        /* Split target dimension by blocks for threaded branches only */
        split_blocks( threadnum,
                      _cd._dimensionSize,
                      _cd._offsetAfter,
                      &blocknum_k,
                      &blocksize_k,
                      &blocksize_last_k );

        /* Each 'i' from _offsetBefore is treated as separate block in threading */
        blocknum_ik = _cd._offsetBefore * blocknum_k;
    }

    /* Compute sums */
    if( do_threading )
    {
        daal::threader_for( blocknum_ik, blocknum_ik, [ & ](int block_ik)
        {
            int block_i = block_ik / blocknum_k;
            int block_k = block_ik % blocknum_k;

            size_t kbeg  = block_k * blocksize_k;
            size_t kend  = kbeg + (( block_k == blocknum_k-1)? blocksize_last_k : blocksize_k);

            _cd.compute_sums((size_t)block_i, kbeg, kend);

        } );
    }
    else
    {
        for (size_t i = 0; i < _cd._offsetBefore; i++)
        {
            _cd.compute_sums(i, 0, _cd._dimensionSize );

        }
    }

    /* Calculate target-dimension output vectors (means, stdev...) */
    if( do_threading && (_cd._dimensionSize >= _MIN_THREAD_SIZE_) )
    {
        daal::threader_for( blocknum_k, blocknum_k, [ & ]( int block_k )
        {
            int ksize = ( block_k == (blocknum_k-1) )? blocksize_last_k : blocksize_k;
            size_t kbeg  = block_k * blocksize_k;
            size_t kend  = kbeg + ksize;

            _cd.compute_means_stdevs( kbeg, kend );

        } );
    }
    else
    {
            _cd.compute_means_stdevs( 0, _cd._dimensionSize );
    }

    /* Calculate "population" vectors if necessary */
    if(parameter->predictionStage == false)
    {
        algorithmFPType alpha = (algorithmFPType)( parameter->alpha );

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for ( size_t k = 0; k < _cd._dimensionSize; k++ )
        {
            _cd._popMean[k]     = _cd._inPopMean[k];
            _cd._popVariance[k] = _cd._inPopVariance[k];
        }

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for ( size_t k = 0; k < _cd._dimensionSize; k++ )
        {
            _cd._popMean[k]     += alpha * _cd._mean[k];
            _cd._popVariance[k] += alpha * _cd._variance[k];
        }
    }

    /* Final gradient result tensor compute */
    if( do_threading )
    {
        daal::threader_for( blocknum_ik, blocknum_ik, [ & ]( int block_ik )
        {
            int block_i = block_ik / blocknum_k;
            int block_k = block_ik % blocknum_k;

            size_t kbeg  = block_k * blocksize_k;
            size_t kend  = kbeg + (( block_k == (blocknum_k-1) )? blocksize_last_k : blocksize_k);

            _cd.compute_gradients( (size_t)block_i, kbeg, kend );

        } );
    }
    else
    {
        for (size_t i = 0; i < _cd._offsetBefore; i++)
        {
            _cd.compute_gradients( i, 0, _cd._dimensionSize );
        }
    }

    DAAL_RETURN_STATUS()
}

} // namespace internal
} // namespace forward
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
