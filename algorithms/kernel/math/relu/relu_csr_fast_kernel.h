/* file: relu_csr_fast_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Declaration of template function that calculate relus.
//--


#ifndef __RELU_CSR_FAST_KERNEL_H__
#define __RELU_CSR_FAST_KERNEL_H__

#include "relu_base.h"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace relu
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
class ReLUKernel<algorithmFPType, fastCSR, cpu> : public ReLUKernelBase<algorithmFPType, fastCSR, cpu>
{
protected:
    Status processBlock(const NumericTable &inputTable, size_t nInputColumns, size_t nProcessedRows, size_t nRowsInCurrentBlock,
                        NumericTable &resultTable);
};

} // namespace daal::internal
} // namespace relu
} // namespace math
} // namespace algorithms
} // namespace daal

#endif
