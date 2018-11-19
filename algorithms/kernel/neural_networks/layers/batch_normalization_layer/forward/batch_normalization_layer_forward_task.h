/* file: batch_normalization_layer_forward_task.h */
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

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_TASK_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_TASK_H__

#include "batch_normalization_layer_forward_task_descriptor.h"

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

using namespace daal::internal;
using namespace daal::services;
using namespace daal::data_management;

#define _MIN_THREAD_SIZE_    16384
#define _MAX_BLOCK_KJ_SIZE_  1048576
#define _MIN_BLOCK_KJ_SIZE_  1024
#define _DEF_BLOCKS_MUL_     2

#define _READ_TENSOR_BLOCK(Type, blockName, tensor, var) \
    Type<algorithmFPType, cpu> blockName(tensor);        \
    DAAL_CHECK_BLOCK_STATUS(blockName);                  \
    var = blockName.get();


template<typename algorithmFPType, Method method, CpuType cpu>
class CommonBatchNormalizationTask
{
private:
    TArray<algorithmFPType, cpu> _invstdwBuffer;
    TArray<algorithmFPType, cpu> _varianceBuffer;
    TArray<algorithmFPType, cpu> _predictionScaleBuffer;
    TArray<algorithmFPType, cpu> _predictionBiasBuffer;

    bool _doThreading;
    bool _predictionStage;
    bool _computeFirstTime;
    bool _hasInputPopMeanAndVariance;

    algorithmFPType _invM;
    algorithmFPType _invM1;
    algorithmFPType _alpha;
    algorithmFPType _epsilon;

    const algorithmFPType *_input;
    const algorithmFPType *_weights;
    const algorithmFPType *_biases;
    const algorithmFPType *_inPopMean;
    const algorithmFPType *_inPopVariance;
    algorithmFPType *_value;
    algorithmFPType *_auxMean;
    algorithmFPType *_auxStd;
    algorithmFPType *_auxPopMean;
    algorithmFPType *_auxPopVariance;
    algorithmFPType *_variance;
    algorithmFPType *_invstdw;
    algorithmFPType *_predictionScale;
    algorithmFPType *_predictionBias;

    size_t _offsetAfter;
    size_t _offsetBefore;
    size_t _dimensionSize;
    size_t _blocknumK;
    size_t _blocksizeK;
    size_t _blocksizeLastK;

public:
    Status initialize(const BatchNormalizationTaskDescriptor &desc)
    {
        _computeFirstTime = true;
        _offsetAfter      = computeTensorOffsetAfterAxis(desc.input, desc.parameter->dimension);
        _offsetBefore     = computeTensorOffsetBeforeAxis(desc.input, desc.parameter->dimension);
        _dimensionSize    = desc.input->getDimensionSize(desc.parameter->dimension);
        _predictionStage  = desc.parameter->predictionStage;

        size_t ij = _offsetBefore * _offsetAfter;
        _invM     = (algorithmFPType)( 1.0 / (double)(ij)     );
        _invM1    = (algorithmFPType)( 1.0 / (double)(ij - 1) );
        _epsilon  = (algorithmFPType)(desc.parameter->epsilon);
        _alpha    = (algorithmFPType)(desc.parameter->alpha);

        if (_predictionStage)
        {
            _predictionScaleBuffer.reset(_dimensionSize); DAAL_CHECK_MALLOC(_predictionScaleBuffer.get());
            _predictionBiasBuffer.reset(_dimensionSize); DAAL_CHECK_MALLOC(_predictionBiasBuffer.get());
            _predictionScale = _predictionScaleBuffer.get();
            _predictionBias = _predictionBiasBuffer.get();

            _READ_TENSOR_BLOCK( ReadSubtensor, weightsBlock,       desc.weights,       _weights       );
            _READ_TENSOR_BLOCK( ReadSubtensor, biasesBlock,        desc.biases,        _biases        );
            _READ_TENSOR_BLOCK( ReadSubtensor, inPopMeanBlock,     desc.inPopMean,     _inPopMean     );
            _READ_TENSOR_BLOCK( ReadSubtensor, inPopVarianceBlock, desc.inPopVariance, _inPopVariance );

            /* sqrt(_inPopVariance[k] + epsilon)                   -> std[k]              */
            /* _weights[k] / std[k]                                -> _predictionScale[k] */
            /* _biases[k] - (_weights[k] * _inPopMean[k]) / std[k] -> _predictionBias[k]  */
            fillScaleAndShiftCoefficients();
        }
        else
        {
            _invstdwBuffer.reset(_dimensionSize);  DAAL_CHECK_MALLOC(_invstdwBuffer.get());
            _varianceBuffer.reset(_dimensionSize); DAAL_CHECK_MALLOC(_varianceBuffer.get());
            _invstdw  = _invstdwBuffer.get();
            _variance = _varianceBuffer.get();
        }

        evaluateNumberOfBlocks();
        return Status();
    }

    Status compute(const BatchNormalizationTaskDescriptor &desc)
    {
        _READ_TENSOR_BLOCK( ReadSubtensor,      inputBlock, desc.input, _input );
        _READ_TENSOR_BLOCK( WriteOnlySubtensor, valueBlock, desc.value, _value );

        if (_predictionStage)
        {
            /* _predictionScale[k] * _input[..., k, ...] + _predictionBias -> _value[..., k, ...] */
            computeValuesOnPredictionStage();
        }
        else
        {
            _READ_TENSOR_BLOCK( ReadSubtensor,      weightsBlock, desc.weights, _weights );
            _READ_TENSOR_BLOCK( ReadSubtensor,      biasesBlock,  desc.biases,  _biases  );
            _READ_TENSOR_BLOCK( WriteOnlySubtensor, auxMeanBlock, desc.auxMean, _auxMean );
            _READ_TENSOR_BLOCK( WriteOnlySubtensor, auxStdBlock,  desc.auxStd,  _auxStd  );

            /* 0 -> _auxMean[k] */
            /* 0 -> _auxStd[k]  */
            fillMeanAndVarianceWithZeros(_auxMean, _auxStd);

            /* _auxMean[k] + _input[..., k, ...]  -> _auxMean[k] */
            /* _auxStd[k] + _input[..., k, ...]^2 -> _auxStd[k]  */
            computeSums();

            /* (_auxStd[k] - _auxMean[k]^2 / M) / (M - 1) -> _variance[k] */
            /* _auxMean[k] / M                            -> _auxMean[k]  */
            /* sqrt(_variance[k] + epsilon)               -> _auxStd[k]   */
            /* _weights[k] / _auxStd[k]                   -> _invstdw[k]  */
            computeMeanAndStd();

            _READ_TENSOR_BLOCK( WriteSubtensor, popMeanBlock,     desc.auxPopMean,     _auxPopMean     );
            _READ_TENSOR_BLOCK( WriteSubtensor, popVarianceBlock, desc.auxPopVariance, _auxPopVariance );

            /* (1 - _alpha) * _auxPopMean[k]     + _alpha * _auxMean[k]  -> _auxPopMean[k]     */
            /* (1 - _alpha) * _auxPopVariance[k] + _alpha * _variance[k] -> _auxPopVariance[k] */
            computePopulationMeanAndVariance();

            /* _invstdw[k] * (_input[..., k, ...] - _auxMean[k]) + _biases[k] -> _value[..., k, ...] */
            computeValuesOnTrainingStage();
        }

        _computeFirstTime = false;
        return Status();
    }


private:
    void computeSums()
    {
        internalThreaderFor([ & ](size_t kbeg, size_t kend)
        {
            for (size_t i = 0; i < _offsetBefore; i++)
            {
                computeSums(i, kbeg, kend);
            }
        });
    }

    inline void computeSums(size_t i_index, size_t kbeg, size_t kend)
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

            _auxMean[k] += sum;
            _auxStd[k]  += sumSq;
        }
    }

    void computeMeanAndStd()
    {
        internalThreaderFor([ & ](size_t kbeg, size_t kend)
        {
            computeMeanAndStd(kbeg, kend);
        }, _dimensionSize >= _MIN_THREAD_SIZE_ /* Threading if true */);
    }

    inline void computeMeanAndStd( size_t kbeg, size_t kend )
    {

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for( size_t k = kbeg; k < kend; k++ )
        {
            _variance[k] = _invM1 * (_auxStd[k] - _auxMean[k] * _auxMean[k] * _invM);
            _auxMean[k] *= _invM;
            _auxStd[k]   = _variance[k] + _epsilon;
        }

        /* Can be replaced to InvSqrt */
        daal::internal::Math<algorithmFPType,cpu>::vSqrt(kend - kbeg, _auxStd + kbeg, _auxStd + kbeg);

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for( size_t k = kbeg; k < kend; k++ )
        {
            _invstdw[k]  = _weights[k] / _auxStd[k];
        }
    }

    void computeValuesOnTrainingStage()
    {
        internalThreaderFor([ & ](size_t kbeg, size_t kend)
        {
            for (size_t i = 0; i < _offsetBefore; i++)
            {
                computeValuesOnTrainingStage(i, kbeg, kend);
            }
        });
    }

    inline void computeValuesOnTrainingStage(size_t i_index, size_t kbeg, size_t kend)
    {
        for(size_t k = kbeg; k < kend; k++)
        {
            size_t idx_ik = ( i_index * _dimensionSize + k ) * _offsetAfter;

            algorithmFPType w = _invstdw[k];
            algorithmFPType m = _auxMean[k];
            algorithmFPType b = _biases[k];

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for ( size_t j = 0; j < _offsetAfter; j++ )
            {
                _value[ idx_ik + j ] = w * ( _input[ idx_ik + j ] - m ) + b;
            }
        }
    }

    void computeValuesOnPredictionStage()
    {
        internalThreaderFor([ & ](size_t kbeg, size_t kend)
        {
            for (size_t i = 0; i < _offsetBefore; i++)
            {
                computeValuesOnPredictionStage(i, kbeg, kend);
            }
        });
    }

    inline void computeValuesOnPredictionStage(size_t i_index, size_t kbeg, size_t kend)
    {
        for(size_t k = kbeg; k < kend; k++)
        {
            size_t idx_ik = ( i_index * _dimensionSize + k ) * _offsetAfter;

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for ( size_t j = 0; j < _offsetAfter; j++ )
            {
                _value[ idx_ik + j ] = _predictionScale[k] * _input[ idx_ik + j ] + _predictionBias[k];
            }
        }
    }

    void computePopulationMeanAndVariance()
    {
        if (_computeFirstTime)
        {
           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for ( size_t k = 0; k < _dimensionSize; k++ )
            {
                _auxPopMean[k]     = _auxMean[k];
                _auxPopVariance[k] = _variance[k];
            }
        }
        else
        {
           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for ( size_t k = 0; k < _dimensionSize; k++ )
            {
                _auxPopMean[k]     += _alpha * (_auxMean[k]  - _auxPopMean[k]);
                _auxPopVariance[k] += _alpha * (_variance[k] - _auxPopVariance[k]);
            }
        }
    }

    void fillScaleAndShiftCoefficients()
    {
        /* Use _predictionScale as temporal buffer to store standard deviation */
       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for ( size_t k = 0; k < _dimensionSize; k++ )
        {
            _predictionScale[k] = _inPopVariance[k] + _epsilon;
        }

        /* Can be replaced to InvSqrt */
        daal::internal::Math<algorithmFPType,cpu>::vSqrt(_dimensionSize, _predictionScale, _predictionScale);

        /* Now _predictionScale stores standard deviation */
       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for ( size_t k = 0; k < _dimensionSize; k++ )
        {
            _predictionScale[k] = _weights[k] / _predictionScale[k];
            _predictionBias[k]  = _biases[k] - _inPopMean[k] * _predictionScale[k];
        }
    }

    void fillMeanAndVarianceWithZeros(algorithmFPType *mean, algorithmFPType *variance)
    {
        for ( size_t i = 0; i < _dimensionSize; i++ )
        {
            mean[i]     = (algorithmFPType)0.0;
            variance[i] = (algorithmFPType)0.0;
        }
    }

    void evaluateNumberOfBlocks()
    {
        size_t ikj       = _offsetBefore * _offsetAfter * _dimensionSize;
        size_t threadnum = services::Environment::getInstance()->getNumberOfThreads();
        _doThreading     = ( ikj > _MIN_THREAD_SIZE_ ) && ( threadnum > 1 );

        if (_doThreading)
        {
            splitBlocks(threadnum, _dimensionSize, _offsetAfter,
                        _blocknumK, _blocksizeK, _blocksizeLastK);
        }
        else
        {
            _blocknumK      = 1;
            _blocksizeK     = _dimensionSize;
            _blocksizeLastK = _dimensionSize;
        }
    }

    void splitBlocks(int threadnum, size_t k_size, size_t j_size,
                     size_t &blocknumK,
                     size_t &blocksizeK,
                     size_t &blocksizeLastK)
    {
        /* Set initial number of blocks by */
        /* k = _DEF_BLOCKS_MUL_ * number_of_threads */
        /* _DEF_BLOCKS_MUL_ > 1 : number of blocks greater than threads */
        blocknumK  = _DEF_BLOCKS_MUL_ * threadnum;

        /* If initial number of blocks greater than k-size, */
        /* set number of blocks = k-size */
        if(blocknumK > k_size)
        {
            blocknumK = k_size;
        }

        /* block size = k-size / number_of_blocks  */
        blocksizeK = k_size / blocknumK;

        /* If block size is too big */
        if( (blocksizeK * j_size) > _MAX_BLOCK_KJ_SIZE_ )
        {
            blocksizeK = _MAX_BLOCK_KJ_SIZE_ / j_size;

            if(blocksizeK < 1)
            {
                blocksizeK = 1;
            }

            blocknumK  = k_size / blocksizeK;
        }
        /* If block size is too small */
        else if( (blocksizeK * j_size) < _MIN_BLOCK_KJ_SIZE_ )
        {
            blocksizeK = _MIN_BLOCK_KJ_SIZE_ / j_size;
            blocknumK  = k_size / blocksizeK;

            if(blocknumK < 1)
            {
                blocknumK = 1;
                blocksizeK = k_size;
            }
        }

        /* Last block size is generally bigger */
        blocksizeLastK = blocksizeK + (k_size - blocknumK * blocksizeK);
    }

    template<typename Operation>
    void internalThreaderFor(Operation operation, bool condition = true)
    {
        if (_doThreading && condition)
        {
            daal::threader_for( _blocknumK, _blocknumK, [ & ]( int block_k )
            {
                size_t kbeg  = block_k * _blocksizeK;
                size_t kend  = kbeg + (( block_k == (_blocknumK-1) ) ? _blocksizeLastK : _blocksizeK);

                operation(kbeg, kend);
            });
        }
        else
        {
            operation(0, _dimensionSize);
        }
    }
};

} // namespace internal
} // namespace forward
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
