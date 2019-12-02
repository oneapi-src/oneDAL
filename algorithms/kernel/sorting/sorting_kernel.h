/* file: sorting_kernel.h */
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
//  Declaration of template structs that sorts data.
//--
*/

#ifndef __SORTING_KERNEL_H__
#define __SORTING_KERNEL_H__

#include "numeric_table.h"
#include "sorting_batch.h"
#include "service_numeric_table.h"
#include "service_stat.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace sorting
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
struct SortingKernel : public Kernel
{
    virtual ~SortingKernel() {}
    Status compute(const NumericTable & inputTable, NumericTable & outputTable);
};

} // namespace internal
} // namespace sorting
} // namespace algorithms
} // namespace daal

#endif
