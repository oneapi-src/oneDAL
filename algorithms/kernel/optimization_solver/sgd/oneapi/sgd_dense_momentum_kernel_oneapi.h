/* file: sgd_dense_momentum_kernel_oneapi.h */
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

/*
//++
//  Implementation of SGD dense momentum Batch Kernel for GPU.
//--
*/

#ifndef __SGD_DENSE_MOMENTUM_KERNEL_ONEAPI_H__
#define __SGD_DENSE_MOMENTUM_KERNEL_ONEAPI_H__

#include "sgd_dense_kernel_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
class SGDKernelOneAPI<algorithmFPType, momentum, cpu> : public Kernel
{
public:
    services::Status compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTablePtr minimum, NumericTable * nIterations,
                             Parameter<momentum> * parameter, NumericTable * learningRateSequence, NumericTable * batchIndices,
                             OptionalArgument * optionalArgument, OptionalArgument * optionalResult, engines::BatchBase & engine)
    {
        return services::ErrorMethodNotImplemented;
    }
};

} // namespace internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
