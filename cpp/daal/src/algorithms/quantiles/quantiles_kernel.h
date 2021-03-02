/* file: quantiles_kernel.h */
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
//  Declaration of template structs that calculate quantiles
//--
*/

#ifndef __QUANTILES_KERNEL_H__
#define __QUANTILES_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/quantiles/quantiles_batch.h"

#include "src/services/service_defines.h"
#include "src/data_management/service_micro_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace quantiles
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
struct QuantilesKernel : public Kernel
{
    virtual ~QuantilesKernel() {}
    services::Status compute(const NumericTable & dataTable, const NumericTable & quantileOrdersTable, NumericTable & quantilesTable);
};

} // namespace internal

} // namespace quantiles

} // namespace algorithms

} // namespace daal

#endif
