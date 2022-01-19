/* file: decision_tree_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for K-Nearest Neighbors training.
//--
*/

#ifndef __DECISION_TREE_REGRESSION_TRAIN_KERNEL_H__
#define __DECISION_TREE_REGRESSION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/decision_tree/decision_tree_regression_training_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;

template <typename algorithmFPType, training::Method method, CpuType cpu>
class DecisionTreeTrainBatchKernel
{};

template <typename algorithmFPType, CpuType cpu>
class DecisionTreeTrainBatchKernel<algorithmFPType, training::defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable * x, const NumericTable * y, const NumericTable * w, const NumericTable * px, const NumericTable * py,
                             decision_tree::regression::Model * r, const daal::algorithms::Parameter * par);
};

} // namespace internal
} // namespace training
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
