/* file: brownboost_predict_kernel.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Declaration of template function that computes Brown Boost predictions.
//--
*/

#ifndef __BROWNBOOST_PREDICT_KERNEL_H__
#define __BROWNBOOST_PREDICT_KERNEL_H__

#include "algorithms/boosting/brownboost_model.h"
#include "algorithms/boosting/brownboost_predict.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace prediction
{
namespace internal
{
template <Method method, typename algorithmFPtype, CpuType cpu>
class BrownBoostPredictKernel : public Kernel
{
public:
    services::Status compute(const NumericTablePtr & x, const Model * m, const NumericTablePtr & r, const Parameter * par);
    services::Status computeImpl(const NumericTablePtr & xTable, const Model * m, size_t nWeakLearners, const algorithmFPtype * alpha,
                                 algorithmFPtype * r, const Parameter * par);
};
} // namespace internal
} // namespace prediction
} // namespace brownboost
} // namespace algorithms
} // namespace daal

#endif
