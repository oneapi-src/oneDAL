/* file: linear_regression_group_of_betas_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of linear regression group of betas quality metrics
//--
*/

#include "src/algorithms/linear_regression/linear_regression_group_of_betas_dense_default_batch_kernel.h"
#include "src/algorithms/linear_regression/linear_regression_group_of_betas_dense_default_batch_impl.i"
#include "src/algorithms/linear_regression/linear_regression_group_of_betas_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace group_of_betas
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
namespace internal
{
template class GroupOfBetasKernel<defaultDense, DAAL_FPTYPE, DAAL_CPU>;
}
} // namespace group_of_betas
} // namespace quality_metric
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
