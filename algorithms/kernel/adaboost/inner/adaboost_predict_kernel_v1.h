/* file: adaboost_predict_kernel_v1.h */
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
//  Declaration of template function that computes Ada Boost predictions.
//--
*/

#ifndef __ADABOOST_PREDICT_KERNEL_H_V1__
#define __ADABOOST_PREDICT_KERNEL_H_V1__

#include "adaboost_model.h"
#include "adaboost_predict.h"
#include "kernel.h"
#include "numeric_table.h"
#include "boosting_predict_kernel.h"

using namespace daal::data_management;
using namespace daal::algorithms::boosting::prediction::internal;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace prediction
{
namespace internal
{
template <Method method, typename algorithmFPtype, CpuType cpu>
class I1AdaBoostPredictKernel : public BoostingPredictKernel<algorithmFPtype, cpu>
{
    using BoostingPredictKernel<algorithmFPtype, cpu>::compute;

public:
    services::Status compute(const NumericTablePtr & x, const daal::algorithms::adaboost::interface1::Model * m, const NumericTablePtr & r,
                             const daal::algorithms::adaboost::interface1::Parameter * par);
};
} // namespace internal
} // namespace prediction
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif
