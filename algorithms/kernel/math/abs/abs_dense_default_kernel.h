/* file: abs_dense_default_kernel.h */
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
//  Declaration of template function that calculate abss.
//--


#ifndef __ABS_DENSE_DEFAULT_KERNEL_H__
#define __ABS_DENSE_DEFAULT_KERNEL_H__

#include "math/abs.h"
#include "kernel.h"
#include "numeric_table.h"

#include "abs_base.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace abs
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
class AbsKernel<algorithmFPType, defaultDense, cpu> : public AbsKernelBase<algorithmFPType, defaultDense, cpu>
{
protected:
    void processBlock(NumericTable* inputTable, size_t nInputColumns, size_t nProcessedRows, size_t nRowsInCurrentBlock,
                      NumericTable* resultTable);
};

} // namespace daal::internal
} // namespace abs
} // namespace math
} // namespace algorithms
} // namespace daal

#endif
