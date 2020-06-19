/* file: smoothrelu_layer_forward_kernel.h */
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
// Implementation of the forward smooth rectifier linear unit (smooth relu) layer
//--

#ifndef __SMOOTHRELU_LAYER_FORWARD_KERNEL_H__
#define __SMOOTHRELU_LAYER_FORWARD_KERNEL_H__

#include "algorithms/neural_networks/layers/smoothrelu/smoothrelu_layer.h"
#include "algorithms/neural_networks/layers/smoothrelu/smoothrelu_layer_types.h"
#include "algorithms/kernel/kernel.h"
#include "externals/service_math.h"
#include "data_management/data/numeric_table.h"
#include "externals/service_blas.h"
#include "algorithms/kernel/neural_networks/layers/layers_threading.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::neural_networks::layers::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace smoothrelu
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for smoothrelu calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class SmoothReLUKernel : public Kernel
{
public:
    services::Status compute(const Tensor & inputTensor, Tensor & resultTensor);
};

} // namespace internal
} // namespace forward
} // namespace smoothrelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
