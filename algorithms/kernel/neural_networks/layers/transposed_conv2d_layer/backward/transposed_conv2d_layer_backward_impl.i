/* file: transposed_conv2d_layer_backward_impl.i */
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

#include "service_dnn.h"
#include "service_dnn_internal.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

#include "transposed_conv2d_layer.h"
#include "transposed_conv2d_layer_types.h"

#include "convolution2d_layer_forward.h"
#include "convolution2d_layer_forward_kernel.h"

#include "convolution2d_layer_backward.h"
#include "convolution2d_layer_backward_kernel.h"

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
namespace transposed_conv2d
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status TransposedConv2dKernel<algorithmFPType, method, cpu>::compute(Tensor *inGradTensor, Tensor *xTensor, Tensor *wTensor,
        const transposed_conv2d::Parameter *parameter, Tensor *wDerTensor, Tensor *bDerTensor, Tensor *resultTensor)
{
    size_t convNKernels = xTensor->getDimensionSize(parameter->groupDimension);
    Collection<size_t> dummyBiasDimensions;
    dummyBiasDimensions.push_back(convNKernels);

    {
        // compute gradient w.r.t. weights
        convolution2d::Parameter bwdConvParameter;
        bwdConvParameter.indices.dims[0] = parameter->indices.dims[0];
        bwdConvParameter.indices.dims[1] = parameter->indices.dims[1];
        bwdConvParameter.nGroups = parameter->nGroups;
        bwdConvParameter.strides.size[0] = parameter->strides.size[0];
        bwdConvParameter.strides.size[1] = parameter->strides.size[1];
        bwdConvParameter.groupDimension = parameter->groupDimension;
        bwdConvParameter.nKernels = convNKernels;
        bwdConvParameter.kernelSizes.size[0] = parameter->kernelSizes.size[0];
        bwdConvParameter.kernelSizes.size[1] = parameter->kernelSizes.size[1];
        bwdConvParameter.paddings.size[0] = parameter->paddings.size[0];
        bwdConvParameter.paddings.size[1] = parameter->paddings.size[1];
        bwdConvParameter.propagateGradient = false;

        Tensor *bwdConvInGradTensor = xTensor;
        Tensor *bwdConvXTensor = inGradTensor;
        Tensor *bwdConvWTensor = wTensor;
        Tensor *bwdConvWDerTensor = wDerTensor;

        convolution2d::backward::internal::Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> dconvKernel;
        dconvKernel.initialize(false, true, false);
        dconvKernel.compute(bwdConvInGradTensor, bwdConvXTensor, bwdConvWTensor, &bwdConvParameter, bwdConvWDerTensor, NULL, NULL);
        dconvKernel.reset();

        if(dconvKernel.getErrorCollection()->size() != 0) {this->_errors->add(dconvKernel.getErrorCollection()); DAAL_RETURN_STATUS();;}
    }

    {
        // compute gradient w.r.t. biases
        const size_t dimsArray[4] = { 0, parameter->groupDimension, parameter->indices.dims[0], parameter->indices.dims[1] };
        TensorOffsetLayout gTargetInLayout = inGradTensor->createDefaultSubtensorLayout();
        gTargetInLayout.shuffleDimensions( services::Collection<size_t>( 4, dimsArray ) );

        ReadSubtensor<algorithmFPType, cpu> inGradBlock(inGradTensor, 0, 0, 0, inGradTensor->getDimensionSize(0), gTargetInLayout);
        algorithmFPType *inGradArray = const_cast<algorithmFPType *>(inGradBlock.get());

        WriteOnlySubtensor<algorithmFPType, cpu> bDerBlock(bDerTensor, 0, 0, 0, bDerTensor->getDimensionSize(0));
        algorithmFPType *bDerArray = bDerBlock.get();

        size_t batchSize = inGradTensor->getDimensionSize(0);
        size_t nKernels = parameter->nKernels;
        size_t channelSize = inGradTensor->getDimensionSize(2) * inGradTensor->getDimensionSize(3);
        for(size_t j = 0; j < nKernels; j++)
        {
            bDerArray[j] = 0;
        }
        for(size_t i = 0; i < batchSize; i++)
        {
            for(size_t j = 0; j < nKernels; j++)
            {
                for(size_t k = 0; k < channelSize; k++)
                {
                    bDerArray[j] += inGradArray[i * nKernels * channelSize + j * channelSize + k];
                }
            }
        }
        algorithmFPType invBatchSize = 1.0 / batchSize;
        for(size_t j = 0; j < nKernels; j++)
        {
            bDerArray[j] *= invBatchSize;
        }
    }

    if(parameter->propagateGradient) // compute gradient w.r.t. data
    {
        convolution2d::Parameter fwdConvParameter;
        fwdConvParameter.indices.dims[0] = parameter->indices.dims[0];
        fwdConvParameter.indices.dims[1] = parameter->indices.dims[1];
        fwdConvParameter.nGroups = parameter->nGroups;
        fwdConvParameter.strides.size[0] = parameter->strides.size[0];
        fwdConvParameter.strides.size[1] = parameter->strides.size[1];
        fwdConvParameter.groupDimension = parameter->groupDimension;
        fwdConvParameter.nKernels = convNKernels;
        fwdConvParameter.kernelSizes.size[0] = parameter->kernelSizes.size[0];
        fwdConvParameter.kernelSizes.size[1] = parameter->kernelSizes.size[1];
        fwdConvParameter.paddings.size[0] = parameter->paddings.size[0];
        fwdConvParameter.paddings.size[1] = parameter->paddings.size[1];

        {
            services::Collection<size_t> inDimsFull = inGradTensor->getDimensions();
            services::Collection<size_t> wDims = wTensor->getDimensions();
            services::Collection<size_t> outDimsFull = resultTensor->getDimensions();

            TArray<algorithmFPType, cpu> bArray(convNKernels);
            HomogenTensor<algorithmFPType> bTensor(dummyBiasDimensions, bArray.get());
            for(size_t i = 0; i < convNKernels; i++) { bArray[i] = 0; }

            convolution2d::forward::internal::Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> convKernel;
            convKernel.initialize(inDimsFull, wDims, &fwdConvParameter, outDimsFull);
            convKernel.compute(inGradTensor, wTensor, &bTensor, &fwdConvParameter, resultTensor);
            convKernel.reset();
            if(convKernel.getErrorCollection()->size() != 0) {this->_errors->add(convKernel.getErrorCollection()); DAAL_RETURN_STATUS();;}
        }
    }
    DAAL_RETURN_STATUS();
}

} // internal
} // backward
} // namespace transposed_conv2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
