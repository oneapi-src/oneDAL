/* file: average_pooling1d_layer_backward_kernel.h */
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
//  Declaration of template function that calculate backward pooling layer relults.
//--


#ifndef __AVERAGE_POOLING1D_LAYER_BACKWARD_KERNEL_H__
#define __AVERAGE_POOLING1D_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/pooling1d/average_pooling1d_layer_backward.h"
#include "neural_networks/layers/pooling1d/average_pooling1d_layer_backward_types.h"
#include "kernel.h"
#include "tensor.h"

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
namespace average_pooling1d
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for backward pooling layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class PoolingKernel : public Kernel
{
public:
    void compute(const average_pooling1d::backward::Input *input,
                const average_pooling1d::Parameter *parameter,
                average_pooling1d::backward::Result *result);
};

} // internal
} // backward
} // average_pooling1d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
