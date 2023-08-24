/* file: logitboost_train_friedman_fpt_cpu.cpp */
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
//  Implementation of Logit Boost training functions.
//--
*/

#include "src/algorithms/logitboost/logitboost_train_batch_container.h"
#include "src/algorithms/logitboost/logitboost_train_friedman_kernel.h"
#include "src/algorithms/logitboost/logitboost_train_friedman_impl.i"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
template class BatchContainer<DAAL_FPTYPE, friedman, DAAL_CPU>;
namespace internal
{
template struct LogitBoostTrainKernel<friedman, DAAL_FPTYPE, DAAL_CPU>;
}
} // namespace training
} // namespace logitboost
} // namespace algorithms
} // namespace daal
