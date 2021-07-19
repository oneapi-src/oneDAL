/* file: gbt_classification_train_kernel.h */
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
//  Declaration of structure containing kernels for gradient boosted trees
//  training.
//--
*/

#ifndef __GBT_CLASSIFICATION_TRAIN_KERNEL_H__
#define __GBT_CLASSIFICATION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_training_types.h"
#include "src/algorithms/engines/engine_batch_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class ClassificationTrainBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(HostAppIface * pHost, const NumericTable * x, const NumericTable * y, gbt::classification::Model & m, Result & res,
                             const interface2::Parameter & par, engines::internal::BatchBaseImpl & engine);
};

} // namespace internal
} // namespace training
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
