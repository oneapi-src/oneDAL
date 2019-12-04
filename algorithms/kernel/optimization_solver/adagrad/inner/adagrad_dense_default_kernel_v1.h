/* file: adagrad_dense_default_kernel_v1.h */
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
//  Declaration of template function that calculate adagrad.
//--

#ifndef __ADAGRAD_DENSE_DEFAULT_KERNEL_V1_H__
#define __ADAGRAD_DENSE_DEFAULT_KERNEL_V1_H__

#include "adagrad_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_math.h"
#include "service_micro_table.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class I1AdagradKernel : public Kernel
{
public:
    services::Status compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTable * minimum, NumericTable * nIterations,
                             NumericTable * gradientSquareSumResult, NumericTable * gradientSquareSumInput, OptionalArgument * optionalArgument,
                             OptionalArgument * optionalResult, interface1::Parameter * parameter, engines::BatchBase & engine);

private:
    static services::Status initAccumulatedGrad(algorithmFPType * accumulatedG, size_t nRows, NumericTable * pOptInput);
    static const size_t _blockSize   = 512;
    static const size_t _threadStart = 50000;
};

} // namespace internal

} // namespace adagrad

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
