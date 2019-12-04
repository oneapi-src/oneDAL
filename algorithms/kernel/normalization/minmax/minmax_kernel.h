/* file: minmax_kernel.h */
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

//++
//  Declaration of template function that calculate minmax.
//--

#ifndef __MINMAX_KERNEL_H__
#define __MINMAX_KERNEL_H__

#include "normalization/minmax.h"
#include "kernel.h"
#include "numeric_table.h"
#include "threading.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"

using namespace daal::services::internal;
using namespace daal::internal;
using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace internal
{
/**
 *  \brief Kernel for minmax calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class MinMaxKernel : public Kernel
{
public:
    Status compute(const NumericTable & inputTable, NumericTable & resultTable, const NumericTable & minimums, const NumericTable & maximums,
                   const algorithmFPType lowerBound, const algorithmFPType upperBound);

protected:
    Status processBlock(const NumericTable & inputTable, NumericTable & resultTable, const algorithmFPType * scale, const algorithmFPType * shift,
                        const size_t startRowIndex, const size_t blockSize);

    static const size_t BLOCK_SIZE_NORM = 256;
};

} // namespace internal
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
