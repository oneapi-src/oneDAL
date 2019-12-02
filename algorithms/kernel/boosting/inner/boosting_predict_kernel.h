/* file: boosting_predict_kernel.h */
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
//  Declaration of template function that computes common boosting predictions.
//--
*/

#ifndef __BOOSTING_PREDICT_KERNEL_H__
#define __BOOSTING_PREDICT_KERNEL_H__

#include "kernel.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace boosting
{
namespace prediction
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
class BoostingPredictKernel : public Kernel
{
protected:
    services::Status compute(const NumericTablePtr & xTable, const Model * m, size_t nWeakLearners, const algorithmFPType * alpha,
                             algorithmFPType * r, const Parameter * par);
};

} // namespace internal
} // namespace prediction
} // namespace boosting
} // namespace algorithms
} // namespace daal

#endif
