/* file: covariance_csr_online_impl.i */
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
services::Status computeSumCSR(
    size_t nFeatures,
    size_t nVectors,
    algorithmFPType *data,
    size_t *colIndices,
    size_t *rowOffsets,
    algorithmFPType *crossProduct,
    algorithmFPType *partialCrossProduct,
    algorithmFPType *sums,
    algorithmFPType *nObservations)
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

    services::Status status = updateCSRCrossProductAndSums<algorithmFPType, method, cpu>(
        nFeatures, nVectors, data, colIndices, rowOffsets, partialCrossProduct, sums, nObservations);
    DAAL_CHECK_STATUS_VAR(status);

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

    return status;
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status computeOtherMethods(
    size_t nFeatures,
    size_t nVectors,
    algorithmFPType *data,
    size_t *colIndices,
    size_t *rowOffsets,
    algorithmFPType *crossProduct,
    algorithmFPType *partialCrossProduct,
    algorithmFPType *sums,
    algorithmFPType *partialSums,
    algorithmFPType *nObservations)
{
    algorithmFPType partialNObservations = 0.0;

    services::Status status = updateCSRCrossProductAndSums<algorithmFPType, method, cpu>(nFeatures, nVectors,
        data, colIndices, rowOffsets, partialCrossProduct, partialSums, &partialNObservations);
    DAAL_CHECK_STATUS_VAR(status);

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

    mergeCrossProductAndSums<algorithmFPType, cpu>(nFeatures, partialCrossProduct,
        partialSums, &partialNObservations, crossProduct, sums, nObservations);

    return status;
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceCSROnlineKernel<algorithmFPType, method, cpu>::compute(
    NumericTable *dataTable,
    NumericTable *nObservationsTable,
    NumericTable *crossProductTable,
    NumericTable *sumTable,
    const Parameter *parameter)
{
    const size_t nFeatures = dataTable->getNumberOfColumns();
    const size_t nVectors  = dataTable->getNumberOfRows();
    CSRNumericTableIface *csrDataTable = dynamic_cast<CSRNumericTableIface* >(dataTable);

    DEFINE_TABLE_BLOCK_EX ( ReadRowsCSR, dataBlock,          csrDataTable,      0, nVectors );
    DEFINE_TABLE_BLOCK    ( WriteRows,   sumBlock,           sumTable                       );
    DEFINE_TABLE_BLOCK    ( WriteRows,   crossProductBlock,  crossProductTable              );
    DEFINE_TABLE_BLOCK    ( WriteRows,   nObservationsBlock, nObservationsTable             );

    algorithmFPType *sums          = sumBlock.get();
    algorithmFPType *crossProduct  = crossProductBlock.get();
    algorithmFPType *nObservations = nObservationsBlock.get();
    algorithmFPType *data          = const_cast<algorithmFPType*>(dataBlock.values());
    size_t          *colIndices    = dataBlock.cols();
    size_t          *rowOffsets    = dataBlock.rows();

    TArrayCalloc<algorithmFPType, cpu> partialCrossProductArray(nFeatures * nFeatures);
    DAAL_CHECK_MALLOC(partialCrossProductArray.get());
    algorithmFPType *partialCrossProduct = partialCrossProductArray.get();

    if (method != sumCSR)
    {
        return computeSumCSR<algorithmFPType, method, cpu>(nFeatures, nVectors, data,
            colIndices, rowOffsets, crossProduct, partialCrossProduct, sums, nObservations);
    }
    else
    {
        NumericTable *userSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
        DEFINE_TABLE_BLOCK( ReadRows, userSumsBlock, userSumsTable );
        algorithmFPType *partialSums = const_cast<algorithmFPType *>(userSumsBlock.get());

        return computeOtherMethods<algorithmFPType, method, cpu>(nFeatures, nVectors, data, colIndices,
            rowOffsets, crossProduct, partialCrossProduct, sums, partialSums, nObservations);
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceCSROnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(
    NumericTable *nObservationsTable,
    NumericTable *crossProductTable,
    NumericTable *sumTable,
    NumericTable *covTable,
    NumericTable *meanTable,
    const Parameter *parameter)
{
    return finalizeCovariance<algorithmFPType, cpu>(nObservationsTable,
        crossProductTable, sumTable, covTable, meanTable, parameter);
}

} // internal
} // covariance
} // algorithms
} // daal

#endif
