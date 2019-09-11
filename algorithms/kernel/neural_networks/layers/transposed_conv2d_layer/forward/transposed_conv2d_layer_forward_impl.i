/* file: transposed_conv2d_layer_forward_impl.i */
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
//  Implementation of transposed_conv2d algorithm
//--
*/

#include "service_tensor.h"
#include "service_numeric_table.h"
#include "convolution2d_layer_backward.h"
#include "convolution2d_layer_backward_kernel.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::algorithms::neural_networks::layers::convolution2d::backward::internal;


namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace transposed_conv2d
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status TransposedConv2dKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, const Tensor &wTensor,
        const Tensor &bTensor, const transposed_conv2d::Parameter &parameter, Tensor &resultTensor)
{
    Status s;
    convolution2d::Parameter bwdConvParameter;
    bwdConvParameter.indices.dims[0] = parameter.indices.dims[0];
    bwdConvParameter.indices.dims[1] = parameter.indices.dims[1];
    bwdConvParameter.nGroups = parameter.nGroups;
    bwdConvParameter.strides.size[0] = parameter.strides.size[0];
    bwdConvParameter.strides.size[1] = parameter.strides.size[1];
    bwdConvParameter.groupDimension = parameter.groupDimension;
    bwdConvParameter.nKernels = inputTensor.getDimensionSize(parameter.groupDimension);
    bwdConvParameter.kernelSizes.size[0] = parameter.kernelSizes.size[0];
    bwdConvParameter.kernelSizes.size[1] = parameter.kernelSizes.size[1];
    bwdConvParameter.paddings.size[0] = parameter.paddings.size[0];
    bwdConvParameter.paddings.size[1] = parameter.paddings.size[1];

    {
        Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> dconvKernel;
        DAAL_CHECK_STATUS(s, dconvKernel.initialize(true, false, false));
        DAAL_CHECK_STATUS(s, dconvKernel.compute(const_cast<Tensor *>(&inputTensor), NULL, const_cast<Tensor *>(&wTensor), bwdConvParameter, NULL, NULL, &resultTensor));
        DAAL_CHECK_STATUS(s, dconvKernel.reset());
    }

    ReadSubtensor<algorithmFPType, cpu> bBlock(const_cast<Tensor &>(bTensor), 0, 0, 0, bTensor.getDimensionSize(0));
    DAAL_CHECK_BLOCK_STATUS(bBlock);
    const algorithmFPType *bArray = bBlock.get();

    WriteSubtensor<algorithmFPType, cpu> resultBlock(resultTensor, 0, 0, 0, resultTensor.getDimensionSize(0));
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType *resultArray = resultBlock.get();

    const size_t batchSize = resultTensor.getDimensionSize(0);
    const size_t nKernels = resultTensor.getDimensionSize(1);
    const size_t channelSize = resultTensor.getDimensionSize(2) * resultTensor.getDimensionSize(3);
    for(size_t i = 0; i < batchSize; i++)
    {
        for(size_t j = 0; j < nKernels; j++)
        {
            for(size_t k = 0; k < channelSize; k++)
            {
                resultArray[i * nKernels * channelSize + j * channelSize + k] += bArray[j];
            }
        }
    }
    return s;
}


} // internal
} // forward
} // namespace transposed_conv2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
