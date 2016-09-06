/* file: covariance_csr_batch_impl.i */
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
//  Covariance matrix computation algorithm implementation in batch mode
//--
*/

#ifndef __COVARIANCE_CSR_BATCH_IMPL_I__
#define __COVARIANCE_CSR_BATCH_IMPL_I__

#include "covariance_kernel.h"
#include "covariance_impl.i"

#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void CovarianceCSRBatchKernel<algorithmFPType, method, cpu>::compute(
            NumericTable *dataTable, NumericTable *covTable,
            NumericTable *meanTable, const Parameter *parameter)
{
    algorithmFPType nObservationsValue = 0.0;
    NumericTablePtr nObservationsTable(
        new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(&nObservationsValue, 1, 1));

    BlockDescriptor<algorithmFPType> crossProductBD, sumBD, nObservationsBD;
    algorithmFPType *crossProduct, *sums, *nObservations;

    size_t nFeatures = dataTable->getNumberOfColumns();
    getDenseCrossProductAndSums<algorithmFPType, method, cpu>(nFeatures, writeOnly,
        covTable, crossProductBD, &crossProduct, meanTable, sumBD, &sums,
        nObservationsTable.get(), nObservationsBD, &nObservations, dataTable);

    algorithmFPType zero = 0.0;
    daal::services::internal::service_memset<algorithmFPType, cpu>(crossProduct, zero, nFeatures * nFeatures);
    if (method != sumCSR)
    {
        daal::services::internal::service_memset<algorithmFPType, cpu>(sums, zero, nFeatures);
    }

    CSRNumericTableIface *csrDataTable = dynamic_cast<CSRNumericTableIface* >(dataTable);

    CSRBlockDescriptor<algorithmFPType> dataBD;
    algorithmFPType *dataBlock;
    size_t *colIndices, *rowOffsets;

    size_t nVectors = dataTable->getNumberOfRows();
    getCSRTableData<algorithmFPType, cpu>(nVectors, readOnly, csrDataTable, dataBD, &dataBlock, &colIndices, &rowOffsets);

    updateCSRCrossProductAndSums<algorithmFPType, method, cpu>(nFeatures, nVectors,
        dataBlock, colIndices, rowOffsets, crossProduct, sums, nObservations, this->_errors.get());

    algorithmFPType invNRows = 1.0 / (algorithmFPType)nVectors;
    for (size_t i = 0; i < nFeatures; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            crossProduct[i * nFeatures + j] -= sums[i] * sums[j] * invNRows;
            crossProduct[j * nFeatures + i]  = crossProduct[i * nFeatures + j];
        }
        crossProduct[i * nFeatures + i] -= sums[i] * sums[i] * invNRows;
    }

    csrDataTable->releaseSparseBlock(dataBD);
    releaseDenseCrossProductAndSums<algorithmFPType, cpu>(covTable, crossProductBD, meanTable, sumBD,
        nObservationsTable.get(), nObservationsBD);

    if (this->_errors->size() != 0) { return; }
    finalizeCovariance<algorithmFPType, cpu>(covTable, meanTable, nObservationsTable.get(), parameter, this->_errors.get());
}

}
}
}
}

#endif
