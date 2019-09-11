/* file: tanh_dense_default_kernel.h */
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
//  Declaration of template function that calculate hyperbolic tangent.
//--


#ifndef __TANH_DENSE_DEFAULT_KERNEL_H__
#define __TANH_DENSE_DEFAULT_KERNEL_H__

#include "tanh_base.h"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace tanh
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
class TanhKernel<algorithmFPType, defaultDense, cpu> : public TanhKernelBase<algorithmFPType, defaultDense, cpu>
{
protected:
    Status processBlock(const NumericTable &inputTable, size_t nInputColumns, size_t nProcessedRows, size_t nRowsInCurrentBlock,
                        NumericTable &resultTable);
};

} // namespace daal::internal
} // namespace tanh
} // namespace math
} // namespace algorithms
} // namespace daal

#endif
