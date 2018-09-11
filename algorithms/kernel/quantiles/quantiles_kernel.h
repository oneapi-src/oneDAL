/* file: quantiles_kernel.h */
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
//  Declaration of template structs that calculate quantiles
//--
*/

#ifndef __QUANTILES_KERNEL_H__
#define __QUANTILES_KERNEL_H__

#include "numeric_table.h"
#include "quantiles_batch.h"

#include "service_defines.h"
#include "service_micro_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace quantiles
{
namespace internal
{

template<Method method, typename algorithmFPType, CpuType cpu>
struct QuantilesKernel : public Kernel
{
    virtual ~QuantilesKernel() {}
    services::Status compute(const NumericTable &dataTable, const NumericTable& quantileOrdersTable, NumericTable &quantilesTable);
};

} // namespace daal::algorithms::quantiles::internal

} // namespace daal::algorithms::quantiles

} // namespace daal::algorithms

} // namespace daal


#endif
