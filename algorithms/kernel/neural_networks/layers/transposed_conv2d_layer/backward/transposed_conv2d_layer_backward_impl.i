/* file: transposed_conv2d_layer_backward_impl.i */
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
services::Status TransposedConv2dKernel<algorithmFPType, method, cpu>::compute(const Tensor &inGradTensor, const Tensor &xTensor, const Tensor &wTensor,
        const transposed_conv2d::Parameter &parameter, Tensor &wDerTensor, Tensor &bDerTensor, Tensor &resultTensor)
{
    Status s;
    const size_t convNKernels = xTensor.getDimensionSize(parameter.groupDimension);

    {
        // compute gradient w.r.t. weights
        convolution2d::Parameter bwdConvParameter;
        bwdConvParameter.indices.dims[0] = parameter.indices.dims[0];
        bwdConvParameter.indices.dims[1] = parameter.indices.dims[1];
        bwdConvParameter.nGroups = parameter.nGroups;
        bwdConvParameter.strides.size[0] = parameter.strides.size[0];
        bwdConvParameter.strides.size[1] = parameter.strides.size[1];
        bwdConvParameter.groupDimension = parameter.groupDimension;
        bwdConvParameter.nKernels = convNKernels;
        bwdConvParameter.kernelSizes.size[0] = parameter.kernelSizes.size[0];
        bwdConvParameter.kernelSizes.size[1] = parameter.kernelSizes.size[1];
        bwdConvParameter.paddings.size[0] = parameter.paddings.size[0];
        bwdConvParameter.paddings.size[1] = parameter.paddings.size[1];
        bwdConvParameter.propagateGradient = false;

        convolution2d::backward::internal::Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> dconvKernel;
        DAAL_CHECK_STATUS(s, dconvKernel.initialize(false, true, false));
        DAAL_CHECK_STATUS(s, dconvKernel.compute(const_cast<Tensor *>(&xTensor), const_cast<Tensor *>(&inGradTensor), const_cast<Tensor *>(&wTensor), bwdConvParameter, &wDerTensor, NULL, NULL));
        DAAL_CHECK_STATUS(s, dconvKernel.reset());
    }

    {
        // compute gradient w.r.t. biases
        const size_t dimsArray[4] = { 0, parameter.groupDimension, parameter.indices.dims[0], parameter.indices.dims[1] };
        TensorOffsetLayout gTargetInLayout = inGradTensor.createDefaultSubtensorLayout();
        gTargetInLayout.shuffleDimensions( services::Collection<size_t>( 4, dimsArray ) );

        ReadSubtensor<algorithmFPType, cpu> inGradBlock(const_cast<Tensor &>(inGradTensor), 0, 0, 0, inGradTensor.getDimensionSize(0), gTargetInLayout);
        DAAL_CHECK_BLOCK_STATUS(inGradBlock);
        const algorithmFPType *inGradArray = inGradBlock.get();

        WriteOnlySubtensor<algorithmFPType, cpu> bDerBlock(bDerTensor, 0, 0, 0, bDerTensor.getDimensionSize(0));
        DAAL_CHECK_BLOCK_STATUS(bDerBlock);
        algorithmFPType *bDerArray = bDerBlock.get();

        const size_t batchSize = inGradTensor.getDimensionSize(0);
        const size_t nKernels = parameter.nKernels;
        const size_t channelSize = inGradTensor.getDimensionSize(2) * inGradTensor.getDimensionSize(3);
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

    if(parameter.propagateGradient) // compute gradient w.r.t. data
    {
        convolution2d::Parameter fwdConvParameter;
        fwdConvParameter.indices.dims[0] = parameter.indices.dims[0];
        fwdConvParameter.indices.dims[1] = parameter.indices.dims[1];
        fwdConvParameter.nGroups = parameter.nGroups;
        fwdConvParameter.strides.size[0] = parameter.strides.size[0];
        fwdConvParameter.strides.size[1] = parameter.strides.size[1];
        fwdConvParameter.groupDimension = parameter.groupDimension;
        fwdConvParameter.nKernels = convNKernels;
        fwdConvParameter.kernelSizes.size[0] = parameter.kernelSizes.size[0];
        fwdConvParameter.kernelSizes.size[1] = parameter.kernelSizes.size[1];
        fwdConvParameter.paddings.size[0] = parameter.paddings.size[0];
        fwdConvParameter.paddings.size[1] = parameter.paddings.size[1];

        {
            const services::Collection<size_t> &inDimsFull = inGradTensor.getDimensions();
            const services::Collection<size_t> &wDims = wTensor.getDimensions();
            const services::Collection<size_t> &outDimsFull = resultTensor.getDimensions();

            TArrayCalloc<algorithmFPType, cpu> bArray(convNKernels);
            DAAL_CHECK_MALLOC(bArray.get());
            Collection<size_t> dummyBiasDimensions;
            dummyBiasDimensions.push_back(convNKernels);
            SharedPtr<HomogenTensor<algorithmFPType> > bTensorPtr = HomogenTensor<algorithmFPType>::create(dummyBiasDimensions, bArray.get(), &s);
            HomogenTensor<algorithmFPType> bTensor = *bTensorPtr;
            DAAL_CHECK_STATUS_VAR(s);

            convolution2d::forward::internal::Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> convKernel;
            DAAL_CHECK_STATUS(s, convKernel.initialize(inDimsFull, wDims, fwdConvParameter, outDimsFull));
            DAAL_CHECK_STATUS(s, convKernel.compute(const_cast<Tensor *>(&inGradTensor), const_cast<Tensor *>(&wTensor), &bTensor, fwdConvParameter, &resultTensor));
            DAAL_CHECK_STATUS(s, convKernel.reset());
        }
    }
    return s;
}

} // internal
} // backward
} // namespace transposed_conv2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
