/* file: logistic_regression_predict_kernel_oneapi.h */
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
//  Declaration of structure containing kernels for logistic regression
//  training.
//--
*/

#ifndef __LOGISTIC_REGRESSION_PREDICT_KERNEL_ONEAPI_H__
#define __LOGISTIC_REGRESSION_PREDICT_KERNEL_ONEAPI_H__

#include "algorithms/logistic_regression/logistic_regression_training_types.h"
#include "common/oneapi/objective_function_utils_oneapi.h"
#include "objective_function/logistic_loss/oneapi/logistic_loss_dense_default_kernel_oneapi.h"
#include "objective_function/cross_entropy_loss/oneapi/cross_entropy_loss_dense_default_kernel_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class PredictBatchKernelOneAPI : public daal::algorithms::Kernel
{
public:
    using LogisticLoss =
        optimization_solver::logistic_loss::internal::LogLossKernelOneAPI<algorithmFPType, optimization_solver::logistic_loss::Method::defaultDense>;

    using CrossEntropyLoss = optimization_solver::cross_entropy_loss::internal::CrossEntropyLossKernelOneAPI<
        algorithmFPType, optimization_solver::cross_entropy_loss::Method::defaultDense>;

    using HelperObjectiveFunction = optimization_solver::objective_function::internal::HelperObjectiveFunction<algorithmFPType>;

    services::Status compute(services::HostAppIface * pHost, NumericTable * x, const logistic_regression::Model * m, size_t nClasses,
                             NumericTable * pRes, NumericTable * pProbab, NumericTable * pLogProbab);

    static services::Status heaviside(const services::Buffer<algorithmFPType> & x, services::Buffer<algorithmFPType> & result, const uint32_t n);

    static services::Status argMax(const services::Buffer<algorithmFPType> & x, services::Buffer<algorithmFPType> & result, const uint32_t n,
                                   const uint32_t p);

private:
    oneapi::internal::UniversalBuffer _fUniversal;
    oneapi::internal::UniversalBuffer _oneVector;
};

} // namespace internal
} // namespace prediction
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

#endif
