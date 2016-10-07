/* file: minmax_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
template<typename algorithmFPType, Method method, CpuType cpu>
class MinMaxKernel : public Kernel
{
public:
    void compute(NumericTable *inputTable, NumericTable *resultTable,
                 NumericTable *minimums, NumericTable *maximums,
                 algorithmFPType lowerBound, algorithmFPType upperBound);

protected:
    void processBlock(NumericTable *inputTable, NumericTable *resultTable, algorithmFPType *scale,
                      algorithmFPType *shift, size_t startRowIndex, size_t blockSize);

    static const size_t BLOCK_SIZE_NORM = 256;
};

} // namespace daal::internal
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
