/* file: pivoted_qr_kernel.h */
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

/*
//++
//  Declaration of template function that calculate qrs.
//--
*/

#ifndef __PIVOTED_QR_KERNEL_H__
#define __PIVOTED_QR_KERNEL_H__

#include "pivoted_qr_batch.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace pivoted_qr
{
namespace internal
{

/**
 *  \brief Kernel for QR calculation
 */
template<daal::algorithms::pivoted_qr::Method method, typename algorithmFPType, CpuType cpu>
class PivotedQRKernel : public Kernel
{
public:
    services::Status compute(const NumericTable &dataTable, NumericTable &QTable, NumericTable &RTable, NumericTable &PTable, NumericTable *permutedColumns);

};

} // namespace daal::internal
}
}
} // namespace daal

#endif
