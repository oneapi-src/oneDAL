/* file: kernel_function_linear_dense_default_batch_oneapi_fpt.cpp */
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
//  Implementation of linear kernel functions for dense input data.
//--
*/

#include "src/algorithms/kernel_function/oneapi/kernel_function_linear_kernel_oneapi.h"
#include "src/algorithms/kernel_function/oneapi/kernel_function_linear_dense_default_oneapi_impl.i"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace internal
{
template class DAAL_EXPORT KernelImplLinearOneAPI<defaultDense, DAAL_FPTYPE>;

} // namespace internal
} // namespace linear
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
