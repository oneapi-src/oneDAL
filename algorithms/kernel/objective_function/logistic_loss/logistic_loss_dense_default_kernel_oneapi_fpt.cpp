/* file: logistic_loss_dense_default_kernel_oneapi_fpt.cpp */
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
//  Implementation of Logistic Loss Batch Kernel for GPU.
//--
*/

#include "oneapi/logistic_loss_dense_default_kernel_oneapi.h"
#include "oneapi/logistic_loss_dense_default_oneapi_impl.i"
#include "logistic_loss_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace logistic_loss
{
namespace internal
{
template class LogLossKernelOneAPI<DAAL_FPTYPE, defaultDense>;

} // namespace internal
} // namespace logistic_loss
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
