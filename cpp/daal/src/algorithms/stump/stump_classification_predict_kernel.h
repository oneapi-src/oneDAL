/* file: stump_classification_predict_kernel.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
//  Declaration of template function that computes Decision Stump predictions.
//--
*/

#ifndef __STUMP_CLASSIFICATION_PREDICT_KERNEL_H__
#define __STUMP_CLASSIFICATION_PREDICT_KERNEL_H__

#include "algorithms/stump/stump_classification_model.h"
#include "algorithms/stump/stump_classification_predict.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace classification
{
namespace prediction
{
namespace internal
{
template <Method method, typename algorithmFPtype, CpuType cpu>
class StumpPredictKernel : public Kernel
{
public:
    services::Status compute(const NumericTable * x, const stump::classification::Model * m, NumericTable * rLabels, NumericTable * rPred,
                             const Parameter * par);

private:
    services::Status changeZeroToMinusOne(NumericTable * yTable);
};

} // namespace internal
} // namespace prediction
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
