/* file: split_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of split algorithm
//--
*/

#ifndef __SPLIT_LAYER_BACKWARD_IMPL_I__
#define __SPLIT_LAYER_BACKWARD_IMPL_I__

#include "threading.h"

#include "service_mkl_tensor.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
Status SplitKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensors[], Tensor *resultTensor, const size_t nInputs)
{
    if (nInputs == 0) { return Status(); }

    MklTensor<algorithmFPType> *resultMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(resultTensor);
    MklTensor<algorithmFPType> *firstInputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputTensors[0]);

    bool canUseMklTensor = (resultMklTensor != 0 && firstInputMklTensor != 0);

    for (size_t i = 1; i < nInputs && canUseMklTensor; i++)
    {
        MklTensor<algorithmFPType> *inputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputTensors[i]);
        if (!inputMklTensor)
        {
            canUseMklTensor = false;
        }
        else if (!dnn::xLayoutCompare((dnnLayout_t)firstInputMklTensor->getDnnLayout(), (dnnLayout_t)inputMklTensor->getDnnLayout()))
        {
            canUseMklTensor = false;
        }
    }

    Status s;
    if (canUseMklTensor)
    {
        const Collection<size_t> &dims = inputTensors[0]->getDimensions();
        size_t dstChannelSize[1] = {dims[1]};
        dnnPrimitive_t splitPrim;
        dnnError_t err;
        dnnLayout_t inputLayout = (dnnLayout_t)firstInputMklTensor->getDnnLayout();
        err = dnn::xSplitCreate(&splitPrim, 1, inputLayout, dstChannelSize); ON_ERR(err);

        dnnLayout_t resultLayout;
        err = dnn::xLayoutCreateFromPrimitive(&resultLayout, splitPrim, dnnResourceSrc); ON_ERR(err);

        resultMklTensor->setDnnLayout(resultLayout);
        const size_t nDataElements = firstInputMklTensor->getSize();
        algorithmFPType *resultArray = resultMklTensor->getDnnArray();
        const algorithmFPType *firstInputArray = firstInputMklTensor->getDnnArray();
        for (size_t i = 0; i < nDataElements; i++)
        {
            resultArray[i] = firstInputArray[i];
        }
        for (size_t i = 1; i < nInputs; i++)
        {
            MklTensor<algorithmFPType> *inputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputTensors[i]);
            algorithmFPType *inputArray = inputMklTensor->getDnnArray();
            for (size_t j = 0; j < nDataElements; j++)
            {
                resultArray[j] += inputArray[j];
            }
        }

        dnn::xDelete(splitPrim);
    }
    else
    {
        const Collection<size_t> &dims = inputTensors[0]->getDimensions();
        const size_t nInputRows = dims[0];

        const size_t nBlocks = nInputRows / _nRowsInBlock;
        const size_t nRowsInLastBlock = nInputRows - nBlocks * _nRowsInBlock;

        for(size_t block = 0; block < nBlocks; block++)
        {
            DAAL_CHECK_STATUS(s, processBlockInit(inputTensors[0], block * _nRowsInBlock, _nRowsInBlock, resultTensor));
        }
        if(nRowsInLastBlock > 0)
        {
            DAAL_CHECK_STATUS(s, processBlockInit(inputTensors[0], nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTensor));
        }

        for(int i = 1; i < nInputs; i++)
        {
            Tensor *inputTensor = inputTensors[i];

            for(size_t block = 0; block < nBlocks; block++)
            {
                DAAL_CHECK_STATUS(s, processBlock(inputTensor, block * _nRowsInBlock, _nRowsInBlock, resultTensor));
            }
            if(nRowsInLastBlock > 0)
            {
                DAAL_CHECK_STATUS(s, processBlock(inputTensor, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTensor));
            }
        }
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline Status SplitKernel<algorithmFPType, method, cpu>::processBlock(Tensor *inputTensor,
                                                                    size_t nProcessedRows,
                                                                    size_t nRowsInCurrentBlock,
                                                                    Tensor *resultTensor)
{
    ReadSubtensor<algorithmFPType, cpu, Tensor> inputSubtensor(inputTensor, 0, 0, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(inputSubtensor);
    const algorithmFPType *inputArray = inputSubtensor.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> resultSubtensor(resultTensor, 0, 0, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(resultSubtensor);
    algorithmFPType *resultArray = resultSubtensor.get();

    const size_t nDataElements = inputSubtensor.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] += inputArray[i];
    }

    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline Status SplitKernel<algorithmFPType, method, cpu>::processBlockInit(Tensor *inputTensor,
                                                                        size_t nProcessedRows,
                                                                        size_t nRowsInCurrentBlock,
                                                                        Tensor *resultTensor)
{
    ReadSubtensor<algorithmFPType, cpu, Tensor> inputSubtensor(inputTensor, 0, 0, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(inputSubtensor);
    const algorithmFPType *inputArray = inputSubtensor.get();

    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultSubtensor(resultTensor, 0, 0, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(resultSubtensor);
    algorithmFPType *resultArray = resultSubtensor.get();

    const size_t nDataElements = inputSubtensor.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] = inputArray[i];
    }

    return Status();
}

} // namespace internal
} // namespace backward
} // namespace split
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
