/* file: covariance_csr_online_impl.i */
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
//  Covariance matrix computation algorithm implementation in distributed mode
//--
*/

#ifndef __COVARIANCE_CSR_ONLINE_IMPL_I__
#define __COVARIANCE_CSR_ONLINE_IMPL_I__

#include "covariance_kernel.h"
#include "covariance_impl.i"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void CovarianceCSROnlineKernel<algorithmFPType, method, cpu>::compute(
            NumericTable *dataTable, NumericTable *nObservationsTable,
            NumericTable *crossProductTable, NumericTable *sumTable,
            const Parameter *parameter)
{
    BlockDescriptor<algorithmFPType> crossProductBD, sumBD, nObservationsBD;
    algorithmFPType *crossProduct, *sums, *nObservations;

    size_t nFeatures = dataTable->getNumberOfColumns();
    getDenseCrossProductAndSums<algorithmFPType, cpu>(readWrite,
        crossProductTable, crossProductBD, &crossProduct, sumTable, sumBD, &sums,
        nObservationsTable, nObservationsBD, &nObservations);

    CSRNumericTableIface *csrDataTable = dynamic_cast<CSRNumericTableIface* >(dataTable);
    CSRBlockDescriptor<algorithmFPType> dataBD;
    algorithmFPType *dataBlock;
    size_t *colIndices, *rowOffsets;

    size_t nVectors = dataTable->getNumberOfRows();
    getCSRTableData<algorithmFPType, cpu>(nVectors, readOnly, csrDataTable, dataBD, &dataBlock, &colIndices, &rowOffsets);

    algorithmFPType *partialCrossProduct;
    partialCrossProduct = (algorithmFPType *)daal_malloc(nFeatures * nFeatures * sizeof(algorithmFPType));
    if (!partialCrossProduct)
    { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    daal::services::internal::service_memset<algorithmFPType, cpu>(partialCrossProduct, (algorithmFPType)0.0, nFeatures * nFeatures);

    if (method != sumCSR)
    {
        algorithmFPType invNObservations = 1.0;
        if (nObservations[0] > 0.5)
        {
            invNObservations = 1.0 / nObservations[0];
            for (size_t i = 0; i < nFeatures; i++)
            {
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] += sums[i] * sums[j] * invNObservations;
                }
            }
        }

        updateCSRCrossProductAndSums<algorithmFPType, method, cpu>(nFeatures, nVectors,
            dataBlock, colIndices, rowOffsets, partialCrossProduct, sums, nObservations, this->_errors.get());

        invNObservations = 1.0 / nObservations[0];
        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
                crossProduct[i * nFeatures + j] -= sums[i] * sums[j] * invNObservations;
                crossProduct[j * nFeatures + i] = crossProduct[i * nFeatures + j];
            }
            crossProduct[i * nFeatures + i] += partialCrossProduct[i * nFeatures + i];
            crossProduct[i * nFeatures + i] -= sums[i] * sums[i] * invNObservations;
        }
    }
    else
    {
        NumericTable *userSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
        if (!userSumsTable) // move to interface check
        { _errors->add(services::ErrorPrecomputedSumNotAvailable); return; }

        BlockDescriptor<algorithmFPType> userSumsBD;
        userSumsTable->getBlockOfRows(0, 1, readOnly, userSumsBD);
        algorithmFPType *partialSums = userSumsBD.getBlockPtr();

        algorithmFPType partialNObservations = 0.0;
        updateCSRCrossProductAndSums<algorithmFPType, method, cpu>(nFeatures, nVectors,
            dataBlock, colIndices, rowOffsets, partialCrossProduct, partialSums, &partialNObservations, this->_errors.get());

        algorithmFPType invPartialNObservations = 1.0 / partialNObservations;
        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                partialCrossProduct[i * nFeatures + j] -= partialSums[i] * partialSums[j] * invPartialNObservations;
                partialCrossProduct[j * nFeatures + i] = partialCrossProduct[i * nFeatures + j];
            }
            partialCrossProduct[i * nFeatures + i] -= partialSums[i] * partialSums[i] * invPartialNObservations;
        }

        mergeCrossProductAndSums<algorithmFPType, cpu>(nFeatures, partialCrossProduct, partialSums,
            &partialNObservations, crossProduct, sums, nObservations);

        userSumsTable->releaseBlockOfRows(userSumsBD);
    }

    daal_free(partialCrossProduct);

    csrDataTable->releaseSparseBlock(dataBD);
    releaseDenseCrossProductAndSums<algorithmFPType, cpu>(crossProductTable, crossProductBD, sumTable, sumBD,
        nObservationsTable, nObservationsBD);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void CovarianceCSROnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(
            NumericTable *nObservationsTable, NumericTable *crossProductTable,
            NumericTable *sumTable, NumericTable *covTable,
            NumericTable *meanTable, const Parameter *parameter)
{
    finalizeCovariance<algorithmFPType, cpu>(crossProductTable, sumTable, nObservationsTable,
        covTable, meanTable, parameter, this->_errors.get());
}

}
}
}
}

#endif
