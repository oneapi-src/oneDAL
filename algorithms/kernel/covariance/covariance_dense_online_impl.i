/* file: covariance_dense_online_impl.i */
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
//  Covariance matrix computation algorithm implementation in distributed mode
//--
*/

#ifndef __COVARIANCE_DENSE_ONLINE_IMPL_I__
#define __COVARIANCE_DENSE_ONLINE_IMPL_I__

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

template<typename algorithmFPType, Method method, CpuType cpu, typename SumsArrayType>
services::Status prepareSums(NumericTable *dataTable,
                             algorithmFPType *&userSums,
                             ReadRows<algorithmFPType, cpu> &userSumsBlock,
                             SumsArrayType &userSumsArray)
{
    if (method == defaultDense)
    {
        userSumsArray.reset(dataTable->getNumberOfColumns());
        DAAL_CHECK_MALLOC(userSumsArray.get());

        userSums = userSumsArray.get();
    }
    else /* method == sumDense */
    {
        NumericTable *userSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
        userSumsBlock.set(userSumsTable, 0, userSumsTable->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(userSumsBlock);

        userSums = const_cast<algorithmFPType *>(userSumsBlock.get());
    }

    return services::Status();
}


template<typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceDenseOnlineKernel<algorithmFPType, method, cpu>::compute(
    NumericTable *dataTable,
    NumericTable *nObservationsTable,
    NumericTable *crossProductTable,
    NumericTable *sumTable,
    const Parameter *parameter)
{
    const size_t nFeatures  = dataTable->getNumberOfColumns();
    const size_t nVectors   = dataTable->getNumberOfRows();
    const bool isNormalized = dataTable->isNormalized(NumericTableIface::standardScoreNormalized);

    DEFINE_TABLE_BLOCK( WriteRows, sumBlock,           sumTable           );
    DEFINE_TABLE_BLOCK( WriteRows, crossProductBlock,  crossProductTable  );
    DEFINE_TABLE_BLOCK( WriteRows, nObservationsBlock, nObservationsTable );
    DEFINE_TABLE_BLOCK( ReadRows,  dataBlock,          dataTable          );

    algorithmFPType *sums          = sumBlock.get();
    algorithmFPType *crossProduct  = crossProductBlock.get();
    algorithmFPType *nObservations = nObservationsBlock.get();
    algorithmFPType *data          = const_cast<algorithmFPType*>(dataBlock.get());

    services::Status status;
    if (method == singlePassDense)
    {
        status |= updateDenseCrossProductAndSums<algorithmFPType, method, cpu>(
            isNormalized, nFeatures, nVectors, data, crossProduct, sums, nObservations);
        DAAL_CHECK_STATUS_VAR(status);
    }
    else
    {
        algorithmFPType partialNObservations = 0.0;

        algorithmFPType                   *userSums;
        ReadRows<algorithmFPType, cpu>     userSumsBlock;
        TArrayCalloc<algorithmFPType, cpu> userSumsArray;

        status |= prepareSums<algorithmFPType, method, cpu>(dataTable, userSums, userSumsBlock, userSumsArray);
        DAAL_CHECK_STATUS_VAR(status);

        TArrayCalloc<algorithmFPType, cpu> partialCrossProductArray(nFeatures * nFeatures);
        DAAL_CHECK_MALLOC(partialCrossProductArray.get());
        algorithmFPType *partialCrossProduct = partialCrossProductArray.get();

        status |= updateDenseCrossProductAndSums<algorithmFPType, method, cpu>(
            isNormalized, nFeatures, nVectors, data, partialCrossProduct, userSums, &partialNObservations);
        DAAL_CHECK_STATUS_VAR(status);

        mergeCrossProductAndSums<algorithmFPType, cpu>(nFeatures, partialCrossProduct, userSums,
            &partialNObservations, crossProduct, sums, nObservations);
    }

    return status;
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceDenseOnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(
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
