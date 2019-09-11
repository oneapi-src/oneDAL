/* file: minmax_kernel.h */
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
template<typename algorithmFPType, Method method, CpuType cpu>
class MinMaxKernel : public Kernel
{
public:
    Status compute(const NumericTable &inputTable, NumericTable &resultTable,
                   const NumericTable &minimums, const NumericTable &maximums,
                   const algorithmFPType lowerBound, const algorithmFPType upperBound);

protected:
    Status processBlock(const NumericTable &inputTable, NumericTable &resultTable,
                        const algorithmFPType *scale, const algorithmFPType *shift,
                        const size_t startRowIndex, const size_t blockSize);

    static const size_t BLOCK_SIZE_NORM = 256;
};

} // namespace daal::internal
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
