/* file: brownboost_predict_dense_default_batch_fpt_cpu_v1.cpp */
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
//  Implementation of Fast method for Brown Boost prediction algorithm.
//--
*/

#include "brownboost_predict_batch_container_v1.h"
#include "brownboost_predict_kernel_v1.h"
#include "brownboost_predict_impl_v1.i"
#include "boosting_predict_impl.i"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace prediction
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}
namespace internal
{
template class I1BrownBoostPredictKernel<defaultDense, DAAL_FPTYPE, DAAL_CPU>;
}
} // namespace prediction
} // namespace brownboost
} // namespace algorithms
} // namespace daal
