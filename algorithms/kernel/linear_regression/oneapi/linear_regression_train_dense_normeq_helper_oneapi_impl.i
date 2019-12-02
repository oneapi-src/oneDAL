/* file: linear_regression_train_dense_normeq_helper_oneapi_impl.i */
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
//  Implementation of auxiliary functions for linear regression
//  Normal Equations (normEqDense) method.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_HELPER_ONEAPI_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_HELPER_ONEAPI_IMPL_I__

#include "linear_regression_train_kernel_oneapi.h"
#include "oneapi/internal/utils.h"
#include "cl_kernel/helper_beta_copy.cl"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{
using namespace daal::oneapi::internal;

template <typename algorithmFPType>
services::Status KernelHelperOneAPI<algorithmFPType>::computeBetasImpl(const size_t p, services::Buffer<algorithmFPType> & a, const size_t ny,
                                                                       services::Buffer<algorithmFPType> & b, const bool inteceptFlag) const
{
    return linear_model::normal_equations::training::internal::FinalizeKernelOneAPI<algorithmFPType>::solveSystem(p, a, ny, b);
}

template <typename algorithmFPType>
services::Status KernelHelperOneAPI<algorithmFPType>::copyBetaToResult(const services::Buffer<algorithmFPType> & betaTmp,
                                                                       services::Buffer<algorithmFPType> & betaRes, const size_t nBetas,
                                                                       const size_t nResponses, const bool interceptFlag) const
{
    services::Status status;
    const size_t nBetasIntercept = interceptFlag ? nBetas : (nBetas - 1);
    const size_t intercept       = interceptFlag ? 1 : 0;

    ExecutionContextIface & ctx    = getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_linear_regression_training_helper_");
    cachekey.add(options);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelHelperBetaCopy, options.c_str());

    const char * const kernelName = "copyBeta";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(5);
    args.set(0, betaTmp, AccessModeIds::read);
    args.set(1, nBetas);
    args.set(2, nBetasIntercept);
    args.set(3, betaRes, AccessModeIds::write);
    args.set(4, intercept);

    KernelRange range(nResponses, nBetas);

    ctx.run(range, kernel, args, &status);

    return status;
}

} // namespace internal
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
