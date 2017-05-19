/* file: logistic_cross_layer_backward_kernel.h */
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
//  Implementation of the backward logistic cross layer
//--


#ifndef __LOGISTIC_CROSS_LAYER_BACKWARD_KERNEL_H__
#define __LOGISTIC_CROSS_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/loss/logistic_cross_layer.h"
#include "neural_networks/layers/loss/logistic_cross_layer_types.h"
#include "neural_networks/layers/loss/logistic_cross_layer_backward_types.h"
#include "kernel.h"
#include "service_math.h"
#include "numeric_table.h"
#include "threading.h"
#include "logistic_layer_forward_kernel.h"

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
namespace loss
{
namespace logistic_cross
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for logistic_cross calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class LogisticCrossKernel : public Kernel
{
public:
    services::Status compute(Tensor *dataTensor, Tensor *groundTruthTensor, const logistic_cross::Parameter *parameter, Tensor *resultTensor);
};

} // namespace internal
} // namespace backward
} // namespace logistic_cross
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
