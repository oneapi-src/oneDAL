/* file: prelu_layer_backward_impl.i */
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
//  Implementation of prelu calculation functions.
//--
*/

#ifndef __PRELU_LAYER_BACKWARD_IMPL_I__
#define __PRELU_LAYER_BACKWARD_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace prelu
{
namespace backward
{
namespace internal
{

using namespace daal::services;

template<typename algorithmFPType, CpuType cpu>
struct Tls_data
{
    Status status;

    TArrayScalableCalloc<algorithmFPType, cpu> wDerBlock;
    TArray<size_t, cpu> fdimsBlock;
    size_t *tls_fDims;
    algorithmFPType* tls_wDerArray;

    DAAL_NEW_DELETE();

    Tls_data(size_t wSize, size_t fDimN) : wDerBlock(wSize), fdimsBlock(fDimN)
    {
        tls_wDerArray = wDerBlock.get();
        if(!tls_wDerArray) { status = Status(ErrorMemoryAllocationFailed); return; }

        tls_fDims = fdimsBlock.get();
        if(!tls_fDims) { status = Status(ErrorMemoryAllocationFailed); return; }

       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < wSize; i++)
        {
            tls_wDerArray[i] = (algorithmFPType)0;
        }
    }

    ~Tls_data() {}
};

template<typename algorithmFPType, Method method, CpuType cpu>
Status PReLUKernel<algorithmFPType, method, cpu>::compute( PReLUTask<algorithmFPType, method, cpu> &task,
                                                         const prelu::Parameter &parameter )
{
    Status s;
    DAAL_CHECK_STATUS(s, task.status);

    if(task.fDimN == 0)
    {
        if (parameter.propagateGradient)
        {
            DAAL_CHECK_STATUS(s, computeGradientBlock(task, 0, task.wDerArray));
        }
        else
        {
            DAAL_CHECK_STATUS(s, computeDerivativesBlock(task, 0, task.wDerArray));
        }
    }
    else
    {
        /* TLS data initialization */
        daal::tls<Tls_data<algorithmFPType, cpu> *> tls_data([ & ]()
        {
            return new Tls_data<algorithmFPType, cpu>(task.wSize, task.fDimN);
        });

        size_t nBlocks = task.xTensor.getSize(0, task.fDimN);

        SafeStatus safeStat;

        if (parameter.propagateGradient)
        {
            __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&(task.inGradTensor)))
            __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&(task.xTensor)))
            __DAAL_MAKE_TENSOR_THREADSAFE(&(task.resultTensor))

            daal::threader_for(nBlocks, nBlocks, [ =, &tls_data, &task, &safeStat ](size_t i)
            {
                Tls_data<algorithmFPType, cpu> *tls_data_local = tls_data.local();
                safeStat |= tls_data_local->status;
                if(!tls_data_local->status) return;

                getFixedDimsIndexes(task.fDimN, tls_data_local->tls_fDims, task.xDims, i);

                safeStat |= computeGradientBlock(task, tls_data_local->tls_fDims, tls_data_local->tls_wDerArray);
            } );
        }
        else
        {
            __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&(task.inGradTensor)))
            __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&(task.xTensor)))

            daal::threader_for(nBlocks, nBlocks, [ =, &tls_data, &task, &safeStat ](size_t i)
            {
                Tls_data<algorithmFPType, cpu> *tls_data_local = tls_data.local();
                safeStat |= tls_data_local->status;
                if(!tls_data_local->status) return;

                getFixedDimsIndexes(task.fDimN, tls_data_local->tls_fDims, task.xDims, i);

                safeStat |= computeDerivativesBlock(task, tls_data_local->tls_fDims, tls_data_local->tls_wDerArray);
            } );
        }

        tls_data.reduce( [ & ]( Tls_data<algorithmFPType, cpu>* tls_data_local )
        {
            if(!safeStat.ok())
            {
                delete tls_data_local;
                return;
            }

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for( size_t i = 0; i < task.wSize; i++)
            {
                task.wDerArray[i] += tls_data_local->tls_wDerArray[i];
            }

            delete tls_data_local;
        } );

        DAAL_CHECK_SAFE_STATUS()
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status PReLUKernel<algorithmFPType, method, cpu>::computeGradientBlock( PReLUTask<algorithmFPType, method, cpu> &task,
                                                                      size_t *fDims,
                                                                      algorithmFPType *wDerArray )
{
    ReadSubtensor<algorithmFPType, cpu> inGradBlock( const_cast<Tensor &>(task.inGradTensor),
                                                     task.fDimN,
                                                     fDims,
                                                     0,
                                                     task.xDims[task.fDimN],
                                                     task.inputLayout );
    DAAL_CHECK_BLOCK_STATUS(inGradBlock);
    const algorithmFPType *inGradArray = inGradBlock.get();

    ReadSubtensor<algorithmFPType, cpu> xBlock( const_cast<Tensor &>(task.xTensor),
                                                task.fDimN,
                                                fDims,
                                                0,
                                                task.xDims[task.fDimN],
                                                task.inputLayout );
    DAAL_CHECK_BLOCK_STATUS(xBlock);
    const algorithmFPType *xArray = xBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> resultBlock( task.resultTensor,
                                                          task.fDimN,
                                                          fDims,
                                                          0,
                                                          task.xDims[task.fDimN],
                                                          task.inputLayout );
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType *resultArray = resultBlock.get();

    size_t nDataElements = xBlock.getSize();
    size_t start = task.wStart;
    size_t end   = task.wStart;

    if(task.wStart + task.wLen <= task.fDimN) // weights are at the left of split dim
    {
        end += task.wLen;
    }
    if( (task.wStart < task.fDimN) &&
        (task.wStart + task.wLen > task.fDimN) ) // split dim is in the midddle of weights dims
    {
        end = task.fDimN;
    }

    size_t wJ = 0;
   PRAGMA_IVDEP
   PRAGMA_VECTOR_ALWAYS
    for(size_t j = start; j < end; j++)
    {
        wJ += fDims[j] * task.wOffsets[j - task.wStart];
    }

    for(size_t i = 0; i < nDataElements; i++)
    {
        if(nDataElements > task.wOffset)
        {
            wJ += (i != 0 && i % task.wOffset == 0);
            if(wJ == task.wSize)
            {
                wJ = 0;
            }
        }

        if (xArray[i] == (algorithmFPType)0)
        {
            resultArray[i] = (algorithmFPType)0;
        }
        else if (xArray[i] > (algorithmFPType)0)
        {
            resultArray[i] = inGradArray[i];
        }
        else
        {
            wDerArray[wJ] += task.invN * inGradArray[i] * xArray[i];
            resultArray[i] = inGradArray[i] * task.wArray[wJ];
        }
    }
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status PReLUKernel<algorithmFPType, method, cpu>::computeDerivativesBlock( PReLUTask<algorithmFPType, method, cpu> &task,
                                                                         size_t *fDims,
                                                                         algorithmFPType *wDerArray )
{
    ReadSubtensor<algorithmFPType, cpu> inGradBlock( const_cast<Tensor &>(task.inGradTensor),
                                                     task.fDimN,
                                                     fDims,
                                                     0,
                                                     task.xDims[task.fDimN],
                                                     task.inputLayout );
    DAAL_CHECK_BLOCK_STATUS(inGradBlock);
    const algorithmFPType *inGradArray = inGradBlock.get();

    ReadSubtensor<algorithmFPType, cpu> xBlock( const_cast<Tensor &>(task.xTensor),
                                                task.fDimN,
                                                fDims,
                                                0,
                                                task.xDims[task.fDimN],
                                                task.inputLayout );
    DAAL_CHECK_BLOCK_STATUS(xBlock);
    const algorithmFPType *xArray = xBlock.get();

    size_t nDataElements = xBlock.getSize();
    size_t start = task.wStart;
    size_t end   = task.wStart;

    if(task.wStart + task.wLen <= task.fDimN) // weights are at the left of split dim
    {
        end += task.wLen;
    }
    if( (task.wStart < task.fDimN) &&
        (task.wStart + task.wLen > task.fDimN) ) // split dim is in the midddle of weights dims
    {
        end = task.fDimN;
    }

    size_t wJ = 0;
    for(size_t j = start; j < end; j++)
    {
        wJ += fDims[j] * task.wOffsets[j - task.wStart];
    }

    for(size_t i = 0; i < nDataElements; i++)
    {
        if(nDataElements > task.wOffset)
        {
            wJ += (i != 0 && i % task.wOffset == 0);
            if(wJ == task.wSize)
            {
                wJ = 0;
            }
        }

        if (xArray[i] < (algorithmFPType)0)
        {
            wDerArray[wJ] += task.invN * inGradArray[i] * xArray[i];
        }
    }
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
PReLUTask<algorithmFPType, method, cpu>::PReLUTask( const Tensor &_inGradTensor,
                                                    const Tensor &_xTensor,
                                                    const Tensor &_wTensor,
                                                    Tensor &_wDerTensor,
                                                    Tensor &_resultTensor,
                                                    const prelu::Parameter &parameter ) : inGradTensor(_inGradTensor),
                                                                                          xTensor(_xTensor),
                                                                                          wTensor(_wTensor),
                                                                                          wDerTensor(_wDerTensor),
                                                                                          resultTensor(_resultTensor),
                                                                                          wDerBlock(),
                                                                                          wBlock(),
                                                                                          wOffsets(parameter.weightsDimension),
                                                                                          inputLayout(_xTensor.createDefaultSubtensorLayout())
{
    wStart  = parameter.dataDimension;
    wLen    = parameter.weightsDimension;
    wSize   = wTensor.getSize();
    fDimN   = 0;
    wOffset = 1;

    invN = 1.0 / (algorithmFPType)(xTensor.getDimensions()[0]);

    xDims = xTensor.getDimensions();

    wDerBlock.set(wDerTensor, 0, 0, 0, wDerTensor.getDimensions()[0]);
    wDerArray = wDerBlock.get();

    wBlock.set(const_cast<Tensor &>(wTensor), 0, 0, 0, wTensor.getDimensions()[0]);
    status |= wBlock.status(); if(!status) return;
    wArray = wBlock.get();

    const Collection<size_t> &wDims = wTensor.getDimensions();

    wOffsets[wLen - 1] = 1;
    for(size_t i = 1; i < wLen; i++)
    {
        wOffsets[wLen - i - 1] = wOffsets[wLen - i] * wDims[wLen - i];
    }

    for(size_t i = 0; i < wSize; i++)
    {
        wDerArray[i] = (algorithmFPType)0;
    }

    size_t nDims = xDims.size();
    Collection<size_t> inputOffsets(nDims);

    inputOffsets[nDims - 1] = 1;
    for(size_t i = 1; i < nDims; i++)
    {
        inputOffsets[nDims - i - 1] = inputOffsets[nDims - i] * xDims[nDims - i];
    }

    wOffset = inputOffsets[wStart + wLen - 1];

    for(int idx = xDims.size() - 1; idx >= 0; idx--)
    {
        if (inputOffsets[idx] > _nElemsInBlock)
        {
            fDimN = idx + 1;
            break;
        }
    }
}

} // namespace internal
} // namespace backward
} // namespace prelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
