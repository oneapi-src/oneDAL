/* file: logitboost_predict_dense_default_fpt_cpu.cpp */
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
//  Implementation of prediction stage of Logit Boost algorithm.
//--
*/

#include "logitboost_predict_batch_container.h"
#include "logitboost_predict_dense_default_kernel.h"
#include "logitboost_predict_dense_default_impl.i"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace prediction
{
namespace interface2
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}
namespace internal
{
template struct LogitBoostPredictKernel<defaultDense, DAAL_FPTYPE, DAAL_CPU>;
}
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal
