/* file: locallyconnected2d_layer_forward_impl.i */
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
services::Status LocallyConnected2dKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, const Tensor &weightsTensor, const Tensor &biasesTensor,
                                Tensor &valueTensor, const locallyconnected2d::Parameter &parameter)
{
    Status s;

    const services::Collection<size_t> &inputDims = inputTensor.getDimensions();
    const services::Collection<size_t> &valueDims = valueTensor.getDimensions();

    size_t nWeightsRows = weightsTensor.getDimensions()[0];
    size_t nBiasesRows  = biasesTensor.getDimensions()[0];

    DAAL_INT m3       = (DAAL_INT)parameter.kernelSizes.size[0];
    DAAL_INT m4       = (DAAL_INT)parameter.kernelSizes.size[1];
    DAAL_INT s3       = (DAAL_INT)parameter.strides.size[0];
    DAAL_INT s4       = (DAAL_INT)parameter.strides.size[1];
    DAAL_INT p3       = (DAAL_INT)parameter.paddings.size[0];
    DAAL_INT p4       = (DAAL_INT)parameter.paddings.size[1];
    DAAL_INT nKernels = (DAAL_INT)parameter.nKernels;
    DAAL_INT nGroups  = (DAAL_INT)parameter.nGroups;

    size_t firstIdx  = parameter.indices.dims[0];
    size_t secondIdx = parameter.indices.dims[1];
    size_t groupDim  = parameter.groupDimension;
    size_t batchDim  = 6 - groupDim - firstIdx - secondIdx;

    size_t n1  = inputDims[batchDim];
    DAAL_INT n2 = (DAAL_INT)inputDims[groupDim];
    DAAL_INT n3 = (DAAL_INT)inputDims[firstIdx];
    DAAL_INT n4 = (DAAL_INT)inputDims[secondIdx];

    DAAL_INT l3 = (DAAL_INT)valueDims[2];
    DAAL_INT l4 = (DAAL_INT)valueDims[3];
    DAAL_INT m2 = n2 / nGroups;
    DAAL_INT nk = nKernels / nGroups;

    const size_t dimsArray[4] = { batchDim, groupDim, firstIdx, secondIdx };

    TensorOffsetLayout inputLayout = inputTensor.createDefaultSubtensorLayout();
    DAAL_CHECK_STATUS(s, inputLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray)));

    ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock;
    WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock;

    ReadSubtensor<algorithmFPType, cpu, Tensor> weightsBlock(const_cast<Tensor &>(weightsTensor), 0, 0, 0, nWeightsRows);
    DAAL_CHECK_BLOCK_STATUS(weightsBlock);
    const algorithmFPType *weightsArray = weightsBlock.get();

    ReadSubtensor<algorithmFPType, cpu, Tensor> biasesBlock(const_cast<Tensor &>(biasesTensor), 0, 0, 0, nBiasesRows);
    DAAL_CHECK_BLOCK_STATUS(biasesBlock);
    const algorithmFPType *biasesArray = biasesBlock.get();

    const algorithmFPType zero = 0.0;

    DAAL_INT resultIndex  = 0;
    DAAL_INT weightsIndex = 0;
    DAAL_INT inputIndex   = 0;
    DAAL_INT biasesIndex  = 0;
    DAAL_INT d = 0;
    DAAL_INT e = 0;

  /*   inputArray  [t] [q * n2 / nGroups + c] [i * s3 - p3 + a] [j * s4 - p4 + b]
       resultArray [t] [q * nKernels / nGroups + r] [i] [j]
       weightsArray    [q * nKernels / nGroups + r] [i] [j] [c] [a] [b]
       biasesArray     [q * nKernels / nGroups + r] [i] [j]
    */
    for(size_t t = 0; t < n1; t++)
    {
        inputBlock.set(const_cast<Tensor &>(inputTensor), 1, &t, 0, (size_t)n2, inputLayout);
        DAAL_CHECK_BLOCK_STATUS(inputBlock);
        const algorithmFPType *inputArray = inputBlock.get();

        resultBlock.set(valueTensor, 1, &t, 0, (size_t)nKernels);
        DAAL_CHECK_BLOCK_STATUS(resultBlock);
        algorithmFPType *resultArray = resultBlock.get();

        for(DAAL_INT q = 0; q < nGroups; q++)
        {
            for(DAAL_INT r = 0; r < nk; r++)
            {
                for(DAAL_INT i = 0; i < l3; i++)
                {
                    for(DAAL_INT j = 0; j < l4; j++)
                    {
                        resultIndex = j + (i + (q * nk + r) * l3) * l4;
                        biasesIndex = j + (i + (q * nk + r) * l3) * l4;
                        resultArray[resultIndex] = biasesArray[biasesIndex];

                        for(DAAL_INT c = 0; c < m2 ; c++)
                        {
                            for(DAAL_INT a = 0; a < m3; a++)
                            {
                                for(DAAL_INT b = 0; b < m4; b++)
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
    return s;
}

} // internal
} // forward
} // namespace locallyconnected2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
