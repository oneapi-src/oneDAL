/* file: quantiles_impl.i */
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

/*
//++
//  Quantiles computation algorithm implementation
//--
*/

#ifndef __QUANTILES_IMPL__
#define __QUANTILES_IMPL__

#include "service_micro_table.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_stat.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace quantiles
{
namespace internal
{
template<Method method, typename AlgorithmFPType, CpuType cpu>
void QuantilesKernel<method, AlgorithmFPType, cpu>::compute(const NumericTable *a, NumericTable *r, const Parameter *par)
{
    size_t nFeatures = a->getNumberOfColumns();
    size_t nVectors = a->getNumberOfRows();
    size_t nQuantileOrders = r->getNumberOfColumns();

    BlockMicroTable<AlgorithmFPType, readOnly, cpu> aMicroTable(const_cast<NumericTable *>(a));
    BlockMicroTable<AlgorithmFPType, writeOnly, cpu> rMicroTable(r);
    BlockMicroTable<AlgorithmFPType, readOnly, cpu> quantsQrderMicroTable(const_cast<NumericTable *>(par->quantileOrders.get()));

    size_t nReadRows = 0;

    AlgorithmFPType *quantileOrders;
    nReadRows = quantsQrderMicroTable.getBlockOfRows(0, 1, &quantileOrders);
    if(nReadRows != 1)
    {
        quantsQrderMicroTable.release();
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    AlgorithmFPType *data;
    nReadRows = aMicroTable.getBlockOfRows(0, nVectors, &data);
    if(nReadRows != nVectors)
    {
        quantsQrderMicroTable.release();
        aMicroTable.release();
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    AlgorithmFPType *quants;
    nReadRows = rMicroTable.getBlockOfRows(0, nFeatures, &quants);
    if(nReadRows != nFeatures)
    {
        aMicroTable.release();
        rMicroTable.release();
        quantsQrderMicroTable.release();
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    int errorcode = Statistics<AlgorithmFPType, cpu>::xQuantiles(data, nFeatures, nVectors, nQuantileOrders, quantileOrders, quants);

    if(errorcode)
    {
        if(errorcode == __DAAL_VSL_SS_ERROR_BAD_QUANT_ORDER) { this->_errors->add(services::ErrorQuantileOrderValueIsInvalid); }
        else { this->_errors->add(services::ErrorQuantilesInternal); }
    }

    aMicroTable.release();
    rMicroTable.release();
    quantsQrderMicroTable.release();
}

} // namespace daal::algorithms::quantiles::internal

} // namespace daal::algorithms::quantiles

} // namespace daal::algorithms

} // namespace daal

#endif
