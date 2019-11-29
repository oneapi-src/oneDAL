/* file: decision_tree_regression_predict_dense_default_batch.h */
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
//  Declaration of template function that computes Decision tree prediction results.
//--
*/

#ifndef __DECISION_TREE_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __DECISION_TREE_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "decision_tree_regression_predict.h"
#include "decision_tree_regression_model_impl.h"
#include "kernel.h"
#include "numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace prediction
{
namespace internal
{
using namespace daal::data_management;

template <typename algorithmFPType, prediction::Method method, CpuType cpu>
class DecisionTreePredictKernel : public daal::algorithms::Kernel
{};

template <typename algorithmFPType, CpuType cpu>
class DecisionTreePredictKernel<algorithmFPType, defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable * x, const daal::algorithms::Model * m, NumericTable * y, const daal::algorithms::Parameter * par);
};

} // namespace internal
} // namespace prediction
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
