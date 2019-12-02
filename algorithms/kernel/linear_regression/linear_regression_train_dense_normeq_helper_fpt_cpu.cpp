/* file: linear_regression_train_dense_normeq_helper_fpt_cpu.cpp */
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
//  Implementation of linear regression training functions for the method
//  of normal equations.
//--
*/

#include "linear_regression_train_dense_normeq_helper_impl.i"

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
template class KernelHelper<DAAL_FPTYPE, DAAL_CPU>;
}
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
