/* file: eltwise_sum_layer_backward_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Declaration of template function that calculate eltwise_sums.
//--


#ifndef __ELTWISE_SUM_LAYER_BACKWARD_KERNEL_H__
#define __ELTWISE_SUM_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/eltwise_sum/eltwise_sum_layer.h"
#include "neural_networks/layers/eltwise_sum/eltwise_sum_layer_types.h"

#include "kernel.h"
#include "layers_threading.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"

using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::neural_networks::layers::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace eltwise_sum
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for eltwise_sum calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class EltwiseSumKernel : public Kernel
{
public:
    services::Status compute(Tensor *inputGradient, Tensor *coefficients,
        Tensor **outputs, size_t nOutputs);

private:
    services::Status processOutputTensor(Tensor *inputGradient,
        const algorithmFPType *coefficientsArray, Tensor *output, size_t outputIndex);

    bool checkForInPlace(const Tensor *inputGradient, const Tensor *coefficients,
        Tensor **outputs, size_t nOutputs);
};

} // namespace internal
} // namespace backward
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
