/* file: df_regression_train_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Declaration of structure containing kernels for decision forest
//  training.
//--
*/

#ifndef __DF_REGRESSION_TRAIN_DENSE_DEFAULT_KERNEL_H__
#define __DF_REGRESSION_TRAIN_DENSE_DEFAULT_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/decision_forest/decision_forest_regression_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
class RegressionTrainBatchKernel<algorithmFPType, defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    services::Status compute(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, decision_forest::regression::Model & m,
                             Result & res, const Parameter & par);
};

} // namespace internal
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
