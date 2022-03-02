/* file: linear_regression_train_dense_normeq_online_oneapi_fpt.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of linear regression training functions for the method
//  of normal equations for GPU in online compute mode.
//--
*/

#include "src/algorithms/linear_regression/oneapi/linear_regression_train_kernel_oneapi.h"
#include "src/algorithms/linear_regression/oneapi/linear_regression_train_dense_normeq_oneapi_impl.i"

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
template class DAAL_EXPORT OnlineKernelOneAPI<DAAL_FPTYPE, normEqDense>;

} // namespace internal
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
