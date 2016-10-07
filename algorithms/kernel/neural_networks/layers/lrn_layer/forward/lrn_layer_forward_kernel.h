/* file: lrn_layer_forward_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of the forward local response normalization layer
//--


#ifndef __LRN_LAYER_FORWARD_KERNEL_H__
#define __LRN_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/lrn/lrn_layer.h"
#include "neural_networks/layers/lrn/lrn_layer_types.h"
#include "kernel.h"
#include "service_math.h"
#include "numeric_table.h"

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
namespace lrn
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for lrn calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class LRNKernel : public Kernel
{
public:
    void compute(Tensor *inputTensor, const lrn::Parameter *parameter, Tensor *sMinusBetaTensor, Tensor *resultTensor);
};
} // internal
} // forward

} // lrn
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
