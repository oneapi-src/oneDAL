/* file: abs_csr_fast_kernel.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

//++
//  Declaration of template function that calculate abss.
//--


#ifndef __ABS_CSR_FAST_KERNEL_H__
#define __ABS_CSR_FAST_KERNEL_H__

#include "abs_base.h"

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
class AbsKernel<algorithmFPType, fastCSR, cpu> : public AbsKernelBase<algorithmFPType, fastCSR, cpu>
{
protected:
    Status processBlock(const NumericTable &inputTable, size_t nInputColumns, size_t nProcessedRows, size_t nRowsInCurrentBlock, NumericTable &resultTable);
};

} // namespace daal::internal
} // namespace abs
} // namespace math
} // namespace algorithms
} // namespace daal

#endif
