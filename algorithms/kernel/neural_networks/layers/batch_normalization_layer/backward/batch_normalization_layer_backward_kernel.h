/* file: batch_normalization_layer_backward_kernel.h */
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
//  Declaration of template function that calculate backward batch normalization layer relults.
//--


#ifndef __BATCH_NORMALIZATION_LAYER_BACKWARD_KERNEL_H__
#define __BATCH_NORMALIZATION_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/batch_normalization/batch_normalization_layer_backward.h"
#include "neural_networks/layers/batch_normalization/batch_normalization_layer_backward_types.h"
#include "kernel.h"
#include "tensor.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for backward batch normalization layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class BatchNormalizationKernel : public Kernel
{
public:
    services::Status compute(Tensor *gradientTensor, Tensor *weightsTensor, Tensor *stDevTensor, Tensor *inputGradientTensor,
                 Tensor *dataTensor, Tensor *meanTensor, Tensor *weightsDerTensor, Tensor *biasesDerTensor,
                 const batch_normalization::Parameter *paramete);

};

} // internal
} // backward
} // batch_normalization
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
