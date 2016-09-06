/* file: adaboost_predict_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#ifndef __ADABOOST_PREDICT_KERNEL_H__
#define __ADABOOST_PREDICT_KERNEL_H__

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
class AdaBoostPredictKernel : public BoostingPredictKernel<algorithmFPtype, cpu>
{
    using BoostingPredictKernel<algorithmFPtype, cpu>::compute;
public:
    void compute(NumericTablePtr x, const Model *m, NumericTablePtr r, const Parameter *par);
};

} // namespace daal::algorithms::adaboost::prediction::internal
}
}
}
} // namespace daal

#endif
