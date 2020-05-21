/* file: df_regression_train_hist_batch_fpt_cpu.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of decision forest regression training functions for the hist method
//--
*/

#include "algorithms/kernel/dtrees/forest/regression/df_regression_train_container.h"
#include "algorithms/kernel/dtrees/forest/regression/df_regression_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, hist, DAAL_CPU>;
}
namespace internal
{
template class RegressionTrainBatchKernel<DAAL_FPTYPE, hist, DAAL_CPU>;
}

} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
