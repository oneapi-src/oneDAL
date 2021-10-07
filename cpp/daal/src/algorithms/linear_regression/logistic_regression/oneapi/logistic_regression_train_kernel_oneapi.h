/* file: logistic_regression_train_kernel_oneapi.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#ifndef __LOGISTIC_REGRESSION_TRAIN_KERNEL_ONEAPI_H__
#define __LOGISTIC_REGRESSION_TRAIN_KERNEL_ONEAPI_H__

#include "algorithms/logistic_regression/logistic_regression_training_types.h"
#include "src/algorithms/objective_function/common/oneapi/objective_function_utils_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method>
class TrainBatchKernelOneAPI : public daal::algorithms::Kernel
{
    using HelperObjectiveFunction = optimization_solver::objective_function::internal::HelperObjectiveFunction<algorithmFPType>;

public:
    services::Status compute(const services::HostAppIfacePtr & pHost, const NumericTablePtr & x, const NumericTablePtr & y,
                             logistic_regression::Model & m, Result & res, const Parameter & par);
};

} // namespace internal
} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

#endif
