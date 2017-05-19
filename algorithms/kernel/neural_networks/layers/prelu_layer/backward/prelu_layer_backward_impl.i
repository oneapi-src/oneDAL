/* file: prelu_layer_backward_impl.i */
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
template<typename algorithmFPType, CpuType cpu>
struct Tls_data
{
    services::ErrorCollection errors;
    TScalableCallocSmartPtr<algorithmFPType, cpu> wDerBlock;
    TArray<size_t, cpu> fdimsBlock;
    size_t *tls_fDims;
    algorithmFPType* tls_wDerArray;

    DAAL_NEW_DELETE();

    Tls_data(size_t wSize, size_t fDimN) : wDerBlock(wSize), fdimsBlock(fDimN)
    {
        tls_wDerArray = wDerBlock.get();
        if(!tls_wDerArray) { errors.add(ErrorMemoryAllocationFailed); return; }

        tls_fDims = fdimsBlock.get();
        if (!tls_fDims) { errors.add(ErrorMemoryAllocationFailed); return; }

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
services::Status PReLUKernel<algorithmFPType, method, cpu>::compute( PReLUTask<algorithmFPType, method, cpu> &task,
                                                         const prelu::Parameter *parameter )
{
    if(task.fDimN == 0)
    {
        if (parameter->propagateGradient)
        {
            computeGradientBlock(task, 0, task.wDerArray);
        }
        else
        {
            computeDerivativesBlock(task, 0, task.wDerArray);
        }
    }
    else
    {
        /* TLS data initialization */
        daal::tls<Tls_data<algorithmFPType, cpu> *> tls_data([ & ]()
        {
            return new Tls_data<algorithmFPType, cpu>(task.wSize, task.fDimN);
        });

        size_t nBlocks = task.xTensor->getSize(0, task.fDimN);

        if (parameter->propagateGradient)
        {
            __DAAL_MAKE_TENSOR_THREADSAFE(task.inGradTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(task.xTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(task.resultTensor)

            daal::threader_for(nBlocks, nBlocks, [ =, &tls_data, &task ](size_t i)
            {
                Tls_data<algorithmFPType, cpu> *tls_data_local = tls_data.local();
                if(tls_data_local->errors.size() != 0) return;

                getFixedDimsIndexes(task.fDimN, tls_data_local->tls_fDims, task.xDims, i);

                computeGradientBlock(task, tls_data_local->tls_fDims, tls_data_local->tls_wDerArray);
            } );
        }
        else
        {
            __DAAL_MAKE_TENSOR_THREADSAFE(task.inGradTensor)
            __DAAL_MAKE_TENSOR_THREADSAFE(task.xTensor)

            daal::threader_for(nBlocks, nBlocks, [ =, &tls_data, &task ](size_t i)
            {
                Tls_data<algorithmFPType, cpu> *tls_data_local = tls_data.local();
                if(tls_data_local->errors.size() != 0) return;

                getFixedDimsIndexes(task.fDimN, tls_data_local->tls_fDims, task.xDims, i);

                computeDerivativesBlock(task, tls_data_local->tls_fDims, tls_data_local->tls_wDerArray);
            } );
        }

        tls_data.reduce( [ & ]( Tls_data<algorithmFPType, cpu>* tls_data_local )
        {
            if(tls_data_local->errors.size() != 0)
            {
                this->_errors->add(ErrorMemoryAllocationFailed);
            }

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for( size_t i = 0; i < task.wSize; i++)
            {
                task.wDerArray[i] += tls_data_local->tls_wDerArray[i];
            }

            delete tls_data_local;
        } );
    }
    DAAL_RETURN_STATUS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PReLUKernel<algorithmFPType, method, cpu>::computeGradientBlock( PReLUTask<algorithmFPType, method, cpu> &task,
                                                                      size_t *fDims,
                                                                      algorithmFPType *wDerArray )
{
    ReadSubtensor<algorithmFPType, cpu> inGradBlock( task.inGradTensor,
                                                     task.fDimN,
                                                     fDims,
                                                     0,
                                                     task.xDims[task.fDimN],
                                                     task.inputLayout );

    const algorithmFPType *inGradArray = inGradBlock.get();

    ReadSubtensor<algorithmFPType, cpu> xBlock( task.xTensor,
                                                task.fDimN,
                                                fDims,
                                                0,
                                                task.xDims[task.fDimN],
                                                task.inputLayout );

    const algorithmFPType *xArray = xBlock.get();

    WriteOnlySubtensor<algorithmFPType, cpu> resultBlock( task.resultTensor,
                                                          task.fDimN,
                                                          fDims,
                                                          0,
                                                          task.xDims[task.fDimN],
                                                          task.inputLayout );

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
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PReLUKernel<algorithmFPType, method, cpu>::computeDerivativesBlock( PReLUTask<algorithmFPType, method, cpu> &task,
                                                                         size_t *fDims,
                                                                         algorithmFPType *wDerArray )
{
    ReadSubtensor<algorithmFPType, cpu> inGradBlock( task.inGradTensor,
                                                     task.fDimN,
                                                     fDims,
                                                     0,
                                                     task.xDims[task.fDimN],
                                                     task.inputLayout );

    const algorithmFPType *inGradArray = inGradBlock.get();

    ReadSubtensor<algorithmFPType, cpu> xBlock( task.xTensor,
                                                task.fDimN,
                                                fDims,
                                                0,
                                                task.xDims[task.fDimN],
                                                task.inputLayout );

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
}

template<typename algorithmFPType, Method method, CpuType cpu>
PReLUTask<algorithmFPType, method, cpu>::PReLUTask( Tensor *_inGradTensor,
                                                    Tensor *_xTensor,
                                                    Tensor *_wTensor,
                                                    Tensor *_wDerTensor,
                                                    Tensor *_resultTensor,
                                                    const prelu::Parameter *parameter ) : wDerBlock(),
                                                                                          wBlock(),
                                                                                          wOffsets(parameter->weightsDimension),
                                                                                          inputLayout(_xTensor->createDefaultSubtensorLayout())
{
    inGradTensor = _inGradTensor;
    xTensor      = _xTensor;
    wTensor      = _wTensor;
    wDerTensor   = _wDerTensor;
    resultTensor = _resultTensor;

    wStart  = parameter->dataDimension;
    wLen    = parameter->weightsDimension;
    wSize   = wTensor->getSize();
    fDimN   = 0;
    wOffset = 1;

    invN = 1.0 / (algorithmFPType)(xTensor->getDimensions()[0]);

    xDims = xTensor->getDimensions();

    wDerBlock.set(*wDerTensor, 0, 0, 0, wDerTensor->getDimensions()[0]);
    wDerArray = wDerBlock.get();

    wBlock.set(*wTensor, 0, 0, 0, wTensor->getDimensions()[0]);
    wArray = wBlock.get();

    const Collection<size_t> &wDims = wTensor->getDimensions();

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
