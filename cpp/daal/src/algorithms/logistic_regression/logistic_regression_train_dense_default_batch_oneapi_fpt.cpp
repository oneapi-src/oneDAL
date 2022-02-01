/* file: logistic_regression_train_dense_default_batch_oneapi_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of Logistic Regression training Batch Kernel for GPU.
//--
*/

#include "src/algorithms/logistic_regression/oneapi/logistic_regression_train_kernel_oneapi.h"
#include "src/algorithms/logistic_regression/oneapi/logistic_regression_train_dense_default_oneapi_impl.i"

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
template class TrainBatchKernelOneAPI<DAAL_FPTYPE, defaultDense>;

} // namespace internal
} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
