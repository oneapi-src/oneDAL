/* file: quantiles_impl.i */
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

/*
//++
//  Quantiles computation algorithm implementation
//--
*/

#ifndef __QUANTILES_IMPL__
#define __QUANTILES_IMPL__

#include "service_numeric_table.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_stat.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace quantiles
{
namespace internal
{
template<Method method, typename algorithmFPType, CpuType cpu>
services::Status QuantilesKernel<method, algorithmFPType, cpu>::compute(
    const NumericTable &dataTable,
    const NumericTable &quantileOrdersTable,
    NumericTable &quantilesTable)
{
    const size_t nFeatures = dataTable.getNumberOfColumns();
    const size_t nVectors = dataTable.getNumberOfRows();
    const size_t nQuantileOrders = quantilesTable.getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> dataBlock(const_cast<NumericTable &>(dataTable), 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(dataBlock)
    const algorithmFPType *data = dataBlock.get();

    ReadRows<algorithmFPType, cpu> quantilesQrderBlock(const_cast<NumericTable &>(quantileOrdersTable), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(quantilesQrderBlock)
    const algorithmFPType *quantileOrders = quantilesQrderBlock.get();

    WriteOnlyRows<algorithmFPType, cpu> quantilesBlock(quantilesTable, 0, nFeatures);
    DAAL_CHECK_BLOCK_STATUS(quantilesBlock)
    algorithmFPType *quantiles = quantilesBlock.get();

    int errorcode = Statistics<algorithmFPType, cpu>::xQuantiles(data, nFeatures, nVectors, nQuantileOrders, quantileOrders, quantiles);

    if(errorcode)
    {
        if(errorcode == __DAAL_VSL_SS_ERROR_BAD_QUANT_ORDER) { return Status(services::ErrorQuantileOrderValueIsInvalid); }
        else { return Status(services::ErrorQuantilesInternal); }
    }

    return Status();
}

} // namespace daal::algorithms::quantiles::internal

} // namespace daal::algorithms::quantiles

} // namespace daal::algorithms

} // namespace daal

#endif
