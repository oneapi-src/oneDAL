/* file: stump_regression_train_kernel.h */
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
//  Declaration of template function that trains Decision Stump.
//--
*/

#ifndef __STUMP_REGRESSION_TRAIN_KERNEL_H__
#define __STUMP_REGRESSION_TRAIN_KERNEL_H__

#include "stump_regression_training_types.h"
#include "stump_regression_model.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
{
namespace training
{
namespace internal
{
template <Method method, typename algorithmFPtype, CpuType cpu>
class StumpTrainKernel : public Kernel
{
public:
    services::Status compute(size_t n, const NumericTable * const * a, Model * r, const Parameter * par);
};

} // namespace internal
} // namespace training
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
