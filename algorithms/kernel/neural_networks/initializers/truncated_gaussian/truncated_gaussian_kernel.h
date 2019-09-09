/* file: truncated_gaussian_kernel.h */
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
//  Declaration of template function that implements truncated gaussian initializer.
//--

#ifndef __TRUNCATED_GAUSSIAN_INITIALIZER_KERNEL_H__
#define __TRUNCATED_GAUSSIAN_INITIALIZER_KERNEL_H__

#include "kernel.h"
#include "service_math.h"
#include "service_tensor.h"
#include "threading.h"
#include "uniform_kernel.h"

#include "neural_networks/initializers/truncated_gaussian/truncated_gaussian_initializer.h"
#include "neural_networks/initializers/truncated_gaussian/truncated_gaussian_initializer_types.h"

#include "truncated_gaussian_task_descriptor.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace truncated_gaussian
{
namespace internal
{

/**
 *  \brief Kernel for truncated_gaussian calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TruncatedGaussianKernel : public Kernel
{
public:
    Status compute(const TruncatedGaussianInitializerTaskDescriptor<algorithmFPType> &desc);

private:
    algorithmFPType getCDFNormal(algorithmFPType p, algorithmFPType mean, algorithmFPType sigma);
    const size_t _nElemsInBlock = 1000;
};

} // internal
} // truncated_gaussian
} // initializers
} // neural_networks
} // algorithms
} // daal

#endif
