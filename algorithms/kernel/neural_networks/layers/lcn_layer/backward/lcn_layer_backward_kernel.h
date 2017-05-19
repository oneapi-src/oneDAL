/* file: lcn_layer_backward_kernel.h */
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

//++
//  Declaration of template function that calculate local contrast normalization.
//--


#ifndef __LCN_LAYER_BACKWARD_KERNEL_H__
#define __LCN_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/lcn/lcn_layer.h"
#include "neural_networks/layers/lcn/lcn_layer_types.h"
#include "convolution2d_layer_backward.h"
#include "kernel.h"
#include "service_math.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
#include "convolution2d_layer_backward_kernel.h"
#include "threading.h"
#include "service_memory.h"
#include "layers_threading.h"

using namespace daal::algorithms::neural_networks::layers::convolution2d::backward::internal;
using namespace daal::data_management;
using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lcn
{
namespace backward
{
namespace internal
{
/**
 *  \brief Kernel for lcn calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class LCNKernel : public Kernel
{
public:
    services::Status compute(Tensor *auxCenteredDataTensor, Tensor *auxSigmaTensor, Tensor *auxCTensor,
                 Tensor *auxInvMaxTensor, Tensor *kernelTensor, Tensor *inGradTensor,
                 Tensor *gradientTensor, const lcn::Parameter *parameter);
    services::Status initialize(Tensor *auxCenteredDataTensor, Tensor *auxSigmaTensor, Tensor *auxCTensor,
                    Tensor *kernelTensor, const lcn::Parameter *parameter);
    services::Status reset();

private:
    size_t nDims;
    size_t nDataRows;
    size_t nSigmaRows;
    size_t nCRows;
    size_t nKernelRows;

    services::Collection<size_t> dataDims;
    services::Collection<size_t> kernelDims;
    Collection<size_t> sigmaDims;

    size_t nDataElements;
    size_t nKernelElements;
    size_t nWeightsElements;
    size_t nCElements;

    size_t dataOffsetBeforeDim;
    size_t dataOffsetAfterDim;

    size_t batchDimension;
    size_t sumDimension;
    size_t firstDim;
    size_t secondDim;

    size_t initialFirstDim;
    size_t initialSecondDim;
    size_t initialSumDimension;
    size_t fDimN;

    double sigmaThreshold;

    convolution2d::Parameter convParameter;

    void getFixedDimsIndexes(size_t *fDims, size_t i);
};

} // internal
} // backward
} // lcn
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
