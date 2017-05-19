/* file: transposed_conv2d_layer_forward_impl.i */
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
services::Status TransposedConv2dKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensor, Tensor *wTensor, Tensor *bTensor,
        const transposed_conv2d::Parameter *parameter, Tensor *resultTensor)
{
    convolution2d::Parameter bwdConvParameter;
    bwdConvParameter.indices.dims[0] = parameter->indices.dims[0];
    bwdConvParameter.indices.dims[1] = parameter->indices.dims[1];
    bwdConvParameter.nGroups = parameter->nGroups;
    bwdConvParameter.strides.size[0] = parameter->strides.size[0];
    bwdConvParameter.strides.size[1] = parameter->strides.size[1];
    bwdConvParameter.groupDimension = parameter->groupDimension;
    bwdConvParameter.nKernels = inputTensor->getDimensionSize(parameter->groupDimension);
    bwdConvParameter.kernelSizes.size[0] = parameter->kernelSizes.size[0];
    bwdConvParameter.kernelSizes.size[1] = parameter->kernelSizes.size[1];
    bwdConvParameter.paddings.size[0] = parameter->paddings.size[0];
    bwdConvParameter.paddings.size[1] = parameter->paddings.size[1];

    Tensor *bwdConvInGradTensor = inputTensor;
    Tensor *bwdConvWTensor = wTensor;
    Tensor *bwdConvResultTensor = resultTensor;

    {
        Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> dconvKernel;
        dconvKernel.initialize(true, false, false);
        dconvKernel.compute(bwdConvInGradTensor, NULL, bwdConvWTensor, &bwdConvParameter, NULL, NULL, bwdConvResultTensor);
        if(dconvKernel.getErrorCollection()->size() != 0) {this->_errors->add(dconvKernel.getErrorCollection()); DAAL_RETURN_STATUS();}
        dconvKernel.reset();
    }

    ReadSubtensor<algorithmFPType, cpu> bBlock(bTensor, 0, 0, 0, bTensor->getDimensionSize(0));
    algorithmFPType *bArray = const_cast<algorithmFPType *>(bBlock.get());

    WriteSubtensor<algorithmFPType, cpu> resultBlock(resultTensor, 0, 0, 0, resultTensor->getDimensionSize(0));
    algorithmFPType *resultArray = resultBlock.get();

    size_t batchSize = resultTensor->getDimensionSize(0);
    size_t nKernels = resultTensor->getDimensionSize(1);
    size_t channelSize = resultTensor->getDimensionSize(2) * resultTensor->getDimensionSize(3);
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
    DAAL_RETURN_STATUS();
}


} // internal
} // forward
} // namespace transposed_conv2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
