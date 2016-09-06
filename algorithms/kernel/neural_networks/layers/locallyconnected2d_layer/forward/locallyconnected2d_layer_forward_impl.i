/* file: locallyconnected2d_layer_forward_impl.i */
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
//  Implementation of locallyconnected2d algorithm
//--
*/

#include "service_blas.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace locallyconnected2d
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void LocallyConnected2dKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, Tensor *weightsTensor, Tensor *biasesTensor,
                                Tensor *valueTensor, const locallyconnected2d::Parameter *parameter)
{
    if(inputTensor == 0 || weightsTensor == 0 || biasesTensor == 0 || valueTensor == 0) { this->_errors->add(services::ErrorNullTensor); return; }

    const services::Collection<size_t> &inputDims = inputTensor->getDimensions();
    const services::Collection<size_t> &valueDims = valueTensor->getDimensions();

    size_t nWeightsRows = weightsTensor->getDimensions()[0];
    size_t nBiasesRows  = biasesTensor->getDimensions()[0];

    MKL_INT m3       = (MKL_INT)parameter->kernelSizes.size[0];
    MKL_INT m4       = (MKL_INT)parameter->kernelSizes.size[1];
    MKL_INT s3       = (MKL_INT)parameter->strides.size[0];
    MKL_INT s4       = (MKL_INT)parameter->strides.size[1];
    MKL_INT p3       = (MKL_INT)parameter->paddings.size[0];
    MKL_INT p4       = (MKL_INT)parameter->paddings.size[1];
    MKL_INT nKernels = (MKL_INT)parameter->nKernels;
    MKL_INT nGroups  = (MKL_INT)parameter->nGroups;

    size_t firstIdx  = parameter->indices.dims[0];
    size_t secondIdx = parameter->indices.dims[1];
    size_t groupDim  = parameter->groupDimension;
    size_t batchDim  = 6 - groupDim - firstIdx - secondIdx;

    size_t n1  = inputDims[batchDim];
    MKL_INT n2 = (MKL_INT)inputDims[groupDim];
    MKL_INT n3 = (MKL_INT)inputDims[firstIdx];
    MKL_INT n4 = (MKL_INT)inputDims[secondIdx];

    MKL_INT l3 = (MKL_INT)valueDims[2];
    MKL_INT l4 = (MKL_INT)valueDims[3];
    MKL_INT m2 = n2 / nGroups;
    MKL_INT nk = nKernels / nGroups;

    const size_t dimsArray[4] = { batchDim, groupDim, firstIdx, secondIdx };

    TensorOffsetLayout inputLayout = inputTensor->createDefaultSubtensorLayout();
    inputLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock;
    WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock;

    ReadSubtensor<algorithmFPType, cpu, Tensor> weightsBlock(*weightsTensor, 0, 0, 0, nWeightsRows);
    const algorithmFPType *weightsArray = weightsBlock.get();

    ReadSubtensor<algorithmFPType, cpu, Tensor> biasesBlock(*biasesTensor, 0, 0, 0, nBiasesRows);
    const algorithmFPType *biasesArray = biasesBlock.get();

    const algorithmFPType zero = 0.0;

    MKL_INT resultIndex  = 0;
    MKL_INT weightsIndex = 0;
    MKL_INT inputIndex   = 0;
    MKL_INT biasesIndex  = 0;
    MKL_INT d = 0;
    MKL_INT e = 0;

  /*   inputArray  [t] [q * n2 / nGroups + c] [i * s3 - p3 + a] [j * s4 - p4 + b]
       resultArray [t] [q * nKernels / nGroups + r] [i] [j]
       weightsArray    [q * nKernels / nGroups + r] [i] [j] [c] [a] [b]
       biasesArray     [q * nKernels / nGroups + r] [i] [j]
    */
    for(size_t t = 0; t < n1; t++)
    {
        inputBlock.set(*inputTensor, 1, &t, 0, (size_t)n2, inputLayout);
        const algorithmFPType *inputArray = inputBlock.get();

        resultBlock.set(*valueTensor, 1, &t, 0, (size_t)nKernels);
        algorithmFPType *resultArray = resultBlock.get();

        for(MKL_INT q = 0; q < nGroups; q++)
        {
            for(MKL_INT r = 0; r < nk; r++)
            {
                for(MKL_INT i = 0; i < l3; i++)
                {
                    for(MKL_INT j = 0; j < l4; j++)
                    {
                        resultIndex = j + (i + (q * nk + r) * l3) * l4;
                        biasesIndex = j + (i + (q * nk + r) * l3) * l4;
                        resultArray[resultIndex] = biasesArray[biasesIndex];

                        for(MKL_INT c = 0; c < m2 ; c++)
                        {
                            for(MKL_INT a = 0; a < m3; a++)
                            {
                                for(MKL_INT b = 0; b < m4; b++)
                                {
                                    weightsIndex = b + (a + (c + (j + (i + (q * nk + r) * l3) * l4) * m2) * m3) * m4;

                                    d = b + j * s4 - p4;
                                    e = a + i * s3 - p3;

                                    if(  d >= 0 && d < n4  &&  e >= 0 && e < n3 )
                                    {
                                        inputIndex = d + (e + (q * m2 + c) * n3) * n4;
                                        resultArray[resultIndex] += weightsArray[weightsIndex] * inputArray[inputIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // internal
} // forward
} // namespace locallyconnected2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
