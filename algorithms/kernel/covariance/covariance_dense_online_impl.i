/* file: covariance_dense_online_impl.i */
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

template<typename algorithmFPType, Method method, CpuType cpu>
void CovarianceDenseOnlineKernel<algorithmFPType, method, cpu>::compute(
            NumericTable *dataTable, NumericTable *nObservationsTable,
            NumericTable *crossProductTable, NumericTable *sumTable,
            const Parameter *parameter)
{
    bool isOnline = true;
    if (method != sumDense)
    {
        updateDensePartialResults<algorithmFPType, method, cpu>(dataTable,
            crossProductTable, sumTable, nObservationsTable, isOnline, this->_errors.get());
    }
    else
    {
        size_t nFeatures = dataTable->getNumberOfColumns();
        size_t nVectors = dataTable->getNumberOfRows();

        BlockDescriptor<algorithmFPType> crossProductBD, sumBD, userSumsBD, nObservationsBD;
        algorithmFPType *crossProduct, *sums, *userSums, *nObservations;

        getDenseCrossProductAndSums<algorithmFPType, cpu>(readWrite,
            crossProductTable, crossProductBD, &crossProduct, sumTable, sumBD, &sums,
            nObservationsTable, nObservationsBD, &nObservations);

        NumericTable *userSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
        if (!userSumsTable) // move to interface check
        { _errors->add(services::ErrorPrecomputedSumNotAvailable); return; }

        userSumsTable->getBlockOfRows(0, userSumsTable->getNumberOfRows(), readOnly, userSumsBD);
        userSums = userSumsBD.getBlockPtr();

        /* Retrieve data associated with input table */
        BlockDescriptor<algorithmFPType> dataBD;
        dataTable->getBlockOfRows(0, nVectors, readOnly, dataBD);
        algorithmFPType *dataBlock = dataBD.getBlockPtr();

        algorithmFPType *partialCrossProduct = (algorithmFPType *)daal_malloc(nFeatures * nFeatures * sizeof(algorithmFPType));
        if (!partialCrossProduct)
        { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

        daal::services::internal::service_memset<algorithmFPType, cpu>(partialCrossProduct, (algorithmFPType)0.0, nFeatures * nFeatures);

        algorithmFPType partialNObservations = 0.0;
        updateDenseCrossProductAndSums<algorithmFPType, method, cpu>(
            dataTable->isNormalized(NumericTableIface::standardScoreNormalized), isOnline,
            nFeatures, nVectors, dataBlock, partialCrossProduct, userSums, &partialNObservations, this->_errors.get());

        mergeCrossProductAndSums<algorithmFPType, cpu>(nFeatures, partialCrossProduct, userSums,
            &partialNObservations, crossProduct, sums, nObservations);

        dataTable->releaseBlockOfRows(dataBD);
        releaseDenseCrossProductAndSums<algorithmFPType, cpu>(crossProductTable, crossProductBD, sumTable, sumBD,
            nObservationsTable, nObservationsBD);
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void CovarianceDenseOnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(
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
