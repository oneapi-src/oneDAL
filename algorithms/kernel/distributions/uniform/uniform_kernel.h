/* file: uniform_kernel.h */
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
//  Declaration of template function that calculates uniform distribution.
//--

#ifndef __UNIFORM_KERNEL_H__
#define __UNIFORM_KERNEL_H__

#include "distributions/uniform/uniform.h"

#include "kernel.h"
#include "numeric_table.h"
#include "engine_batch_impl.h"

#include "service_rng.h"
#include "service_unique_ptr.h"
#include "service_numeric_table.h"

using namespace daal::services;
using namespace daal::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace uniform
{
namespace internal
{
/**
 *  \brief Kernel for uniform calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class UniformKernel : public Kernel
{
public:
    static Status compute(const uniform::Parameter<algorithmFPType> & parameter, engines::BatchBase & engine, NumericTable * resultTable);
    static Status compute(const uniform::Parameter<algorithmFPType> & parameter, engines::BatchBase & engine, size_t n,
                          algorithmFPType * resultArray);
    static Status compute(const uniform::Parameter<algorithmFPType> & parameter, UniquePtr<engines::internal::BatchBaseImpl, cpu> & enginePtr,
                          size_t n, algorithmFPType * resultArray);
    static Status compute(algorithmFPType a, algorithmFPType b, engines::BatchBase & engine, size_t n, algorithmFPType * resultArray);
    static Status compute(algorithmFPType a, algorithmFPType b, engines::internal::BatchBaseImpl & engine, size_t n, algorithmFPType * resultArray);
};

template <typename algorithmFPType, CpuType cpu>
using UniformKernelDefault = UniformKernel<algorithmFPType, uniform::defaultDense, cpu>;

} // namespace internal
} // namespace uniform
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
