/* file: quantiles_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Quantiles computation algorithm implementation
//--
*/

#ifndef __QUANTILES_IMPL__
#define __QUANTILES_IMPL__

#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/externals/service_stat.h"

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
template <Method method, typename algorithmFPType, CpuType cpu>
services::Status QuantilesKernel<method, algorithmFPType, cpu>::compute(const NumericTable & dataTable, const NumericTable & quantileOrdersTable,
                                                                        NumericTable & quantilesTable)
{
    const size_t nFeatures       = dataTable.getNumberOfColumns();
    const size_t nVectors        = dataTable.getNumberOfRows();
    const size_t nQuantileOrders = quantilesTable.getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> dataBlock(const_cast<NumericTable &>(dataTable), 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(dataBlock)
    const algorithmFPType * data = dataBlock.get();

    ReadRows<algorithmFPType, cpu> quantilesQrderBlock(const_cast<NumericTable &>(quantileOrdersTable), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(quantilesQrderBlock)
    const algorithmFPType * quantileOrders = quantilesQrderBlock.get();

    WriteOnlyRows<algorithmFPType, cpu> quantilesBlock(quantilesTable, 0, nFeatures);
    DAAL_CHECK_BLOCK_STATUS(quantilesBlock)
    algorithmFPType * quantiles = quantilesBlock.get();

    int errorcode = StatisticsInst<algorithmFPType, cpu>::xQuantiles(data, nFeatures, nVectors, nQuantileOrders, quantileOrders, quantiles);

    if (errorcode)
    {
        if (errorcode == __DAAL_VSL_SS_ERROR_BAD_QUANT_ORDER)
        {
            return Status(services::ErrorQuantileOrderValueIsInvalid);
        }
        else
        {
            return Status(services::ErrorQuantilesInternal);
        }
    }

    return Status();
}

} // namespace internal

} // namespace quantiles

} // namespace algorithms

} // namespace daal

#endif
