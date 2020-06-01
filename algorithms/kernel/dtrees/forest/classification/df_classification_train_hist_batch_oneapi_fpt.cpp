/* file: df_classification_train_hist_batch_oneapi_fpt.cpp */
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
//  Implementation of decision forest classification training functions for the hist method
//--
*/

#include "algorithms/kernel/dtrees/forest/classification/oneapi/df_classification_train_hist_kernel_oneapi.h"
#include "algorithms/kernel/dtrees/forest/classification/oneapi/df_classification_train_hist_oneapi_impl.i"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace internal
{
template class ClassificationTrainBatchKernelOneAPI<DAAL_FPTYPE, hist>;
}

} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
