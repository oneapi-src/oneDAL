/* file: transposed_conv2d_layer_forward_kernel.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Declaration of template function that calculate transposed convolution 2d.
//--

#ifndef __TRANSPOSED_CONV2D_LAYER_FORWARD_KERNEL_H__
#define __TRANSPOSED_CONV2D_LAYER_FORWARD_KERNEL_H__

#include "algorithms/neural_networks/layers/transposed_conv2d/transposed_conv2d_layer.h"
#include "algorithms/neural_networks/layers/transposed_conv2d/transposed_conv2d_layer_types.h"
#include "algorithms/kernel/kernel.h"
#include "externals/service_math.h"
#include "data_management/data/numeric_table.h"
#include "externals/service_dnn.h"
#include "algorithms/kernel/service_dnn_internal.h"

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
namespace transposed_conv2d
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for transposed convolution 2d calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class TransposedConv2dKernel : public Kernel
{
public:
    TransposedConv2dKernel() {}

    services::Status compute(const Tensor & inputTensor, const Tensor & wTensor, const Tensor & bTensor,
                             const transposed_conv2d::Parameter & parameter, Tensor & resultTensor);
};
} // namespace internal
} // namespace forward

} // namespace transposed_conv2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
