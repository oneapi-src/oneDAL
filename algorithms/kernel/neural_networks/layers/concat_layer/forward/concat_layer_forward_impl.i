/* file: concat_layer_forward_impl.i */
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
//  Implementation of concat algorithm
//--
*/

#include "service_mkl_tensor.h"

#ifndef __CONCAT_LAYER_FORWARD_IMPL_I__
#define __CONCAT_LAYER_FORWARD_IMPL_I__

using namespace daal::internal;
using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace concat
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
Status ConcatKernel<algorithmFPType, method, cpu>::compute(size_t nInputs, Tensor *inputTensors[], const concat::Parameter *parameter,
                                                         Tensor *resultTensor)
{
    if (nInputs == 0) { return Status(); }

    const size_t concatDimension = parameter->concatDimension;

    const size_t nDims = resultTensor->getNumberOfDimensions();

    MklTensor<algorithmFPType> *resultMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(resultTensor);

    bool canUseMklTensor = (resultMklTensor != 0 && concatDimension == 1 && (nDims == 4 || nDims == 5));

    for (size_t i = 0; i < nInputs && canUseMklTensor; i++)
    {
        MklTensor<algorithmFPType> *inputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputTensors[i]);
        if (!inputMklTensor)
        {
            canUseMklTensor = false;
        }
    }

    if (canUseMklTensor)
    {
        algorithmFPType* concatRes[dnnResourceNumber] = {0};

        dnnLayout_t resultLayout;
        dnnError_t err;

        if (concatPrim == NULL)
        {
            if (inputLayouts)
            {
                delete [] inputLayouts;
            }
            inputLayouts = new dnnLayout_t[nInputs];
            for (size_t i = 0; i < nInputs; i++)
            {
                MklTensor<algorithmFPType> *inputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputTensors[i]);
                inputLayouts[i] = (dnnLayout_t)inputMklTensor->getDnnLayout();
            }
            err = dnn::xConcatCreate( &concatPrim, nInputs, inputLayouts); ON_ERR(err);
        }

        for (size_t i = 0; i < nInputs; i++)
        {
            MklTensor<algorithmFPType> *inputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputTensors[i]);
            concatRes[dnnResourceMultipleSrc + i] = inputMklTensor->getDnnArray();
        }

        err = dnn::xLayoutCreateFromPrimitive(&resultLayout, concatPrim, dnnResourceDst); ON_ERR(err);
        resultMklTensor->setDnnLayout(resultLayout);

        concatRes[dnnResourceDst] = resultMklTensor->getDnnArray();

        err = dnn::xExecute(concatPrim, (void**)concatRes); ON_ERR(err);
    }
    else
    {
        const Collection<size_t> &resDims = resultTensor->getDimensions();
        const size_t nResultRows = resDims[0];

        WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultSubtensor(resultTensor, 0, 0, 0, nResultRows);
        DAAL_CHECK_BLOCK_STATUS(resultSubtensor);
        algorithmFPType *resultArray = resultSubtensor.get();

        const size_t dimsSum = resDims[concatDimension];

        size_t offsetBefore = 1;
        for(size_t j = 0; j < concatDimension; j++)
        {
            offsetBefore *= resDims[j];
        }

        size_t offsetAfter = 1;
        for(size_t j = concatDimension + 1; j < resDims.size(); j++)
        {
            offsetAfter *= resDims[j];
        }

        size_t sum = 0;
        for(size_t l = 0; l < nInputs; l++)
        {
            Tensor *inputTensor = inputTensors[l];

            const Collection<size_t> &dims = inputTensor->getDimensions();
            const size_t nInputRows = dims[0];

            ReadSubtensor<algorithmFPType, cpu, Tensor> inputSubtensor(inputTensor, 0, 0, 0, nInputRows);
            DAAL_CHECK_BLOCK_STATUS(inputSubtensor);
            const algorithmFPType *inputArray = inputSubtensor.get();

            for(size_t i = 0; i < offsetBefore; i++)
            {

                for(size_t k = 0; k < dims[concatDimension]; k++)
                {
                    for(size_t j = 0; j < offsetAfter; j++)
                    {
                        const size_t inputIndex = (i * dims[concatDimension] + k) * offsetAfter + j;
                        const size_t outputIndex = (i * dimsSum + k + sum) * offsetAfter + j;
                        resultArray[outputIndex] = inputArray[inputIndex];
                    }
                }
            }

            sum += dims[concatDimension];
        }

    }
    return Status();
}

} // namespace internal
} // namespace forward
} // namespace concat
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
