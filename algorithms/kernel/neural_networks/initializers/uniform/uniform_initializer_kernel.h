/* file: uniform_initializer_kernel.h */
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
//  Declaration of template function that calculate uniforms.
//--


#ifndef __UNIFORM_INITIALIZER_KERNEL_H__
#define __UNIFORM_INITIALIZER_KERNEL_H__

#include "neural_networks/initializers/uniform/uniform_initializer.h"
#include "neural_networks/initializers/uniform/uniform_initializer_types.h"
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
namespace initializers
{
namespace uniform
{
namespace internal
{

/**
 *  \brief Kernel for uniform calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class UniformKernel : public Kernel
{
public:
    void compute(const initializers::Input *input, const uniform::Parameter *parameter,
                 initializers::Result *result);
};

} // internal
} // uniform
} // initializers
} // neural_networks
} // algorithms
} // daal

#endif
