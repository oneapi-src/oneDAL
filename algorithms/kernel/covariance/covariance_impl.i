/* file: covariance_impl.i */
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
//  Covariance matrix computation algorithm implementation
//--
*/

#ifndef __COVARIANCE_IMPL_I__
#define __COVARIANCE_IMPL_I__

#include "numeric_table.h"
#include "csr_numeric_table.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_blas.h"
#include "service_spblas.h"
#include "service_stat.h"
#include "threading.h"



using namespace daal::internal;
using namespace daal::services::internal;


namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{

/*********************** getTableData ************************************************************/
template<typename algorithmFPType, CpuType cpu>
void getTableData(ReadWriteMode                   rwMode,
                 NumericTable                     *numericTable,
                 BlockDescriptor<algorithmFPType> &bd,
                 algorithmFPType                  **dataArray)
{
    size_t nRows = numericTable->getNumberOfRows();
    numericTable->getBlockOfRows(0, nRows, rwMode, bd);
    *dataArray = bd.getBlockPtr();
}

/************************* getCSRTableData ********************************************************/
template<typename algorithmFPType, CpuType cpu>
void getCSRTableData(size_t                              nRows,
                     ReadWriteMode                       rwMode,
                     CSRNumericTableIface                *numericTable,
                     CSRBlockDescriptor<algorithmFPType> &bd,
                     algorithmFPType                     **dataArray,
                     size_t                              **colIndices,
                     size_t                              **rowOffsets)
{
    numericTable->getSparseBlock(0, nRows, rwMode, bd);
    *dataArray  = bd.getBlockValuesPtr();
    *colIndices = bd.getBlockColumnIndicesPtr();
    *rowOffsets = bd.getBlockRowIndicesPtr();
}

/****************************** getDenseCrossProductAndSums ***************************************/
template<typename algorithmFPType, CpuType cpu>
void getDenseCrossProductAndSums(ReadWriteMode                    rwMode,
                                 NumericTable                     *covTable,
                                 BlockDescriptor<algorithmFPType> &crossProductBD,
                                 algorithmFPType                  **crossProduct,
                                 NumericTable                     *meanTable,
                                 BlockDescriptor<algorithmFPType> &sumBD,
                                 algorithmFPType                  **sums,
                                 NumericTable                     *nObservationsTable,
                                 BlockDescriptor<algorithmFPType> &nObservationsBD,
                                 algorithmFPType                  **nObservations)
{
    getTableData<algorithmFPType, cpu>(rwMode, covTable,           crossProductBD,  crossProduct);
    getTableData<algorithmFPType, cpu>(rwMode, meanTable,          sumBD,           sums);
    getTableData<algorithmFPType, cpu>(rwMode, nObservationsTable, nObservationsBD, nObservations);
}

/************************** getDenseCrossProductAndSums *******************************************/
template<typename algorithmFPType, Method method, CpuType cpu>
void getDenseCrossProductAndSums(size_t                            nFeatures,
                                 ReadWriteMode                     rwMode,
                                 NumericTable                      *covTable,
                                 BlockDescriptor<algorithmFPType>  &crossProductBD,
                                 algorithmFPType                   **crossProduct,
                                 NumericTable                      *meanTable,
                                 BlockDescriptor<algorithmFPType>  &sumBD,
                                 algorithmFPType                   **sums,
                                 NumericTable                      *nObservationsTable,
                                 BlockDescriptor<algorithmFPType>  &nObservationsBD,
                                 algorithmFPType                   **nObservations,
                                 NumericTable                      *dataTable)
{
    getDenseCrossProductAndSums<algorithmFPType, cpu>(rwMode,
                                                      covTable,
                                                      crossProductBD,
                                                      crossProduct,
                                                      meanTable,
                                                      sumBD,
                                                      sums,
                                                      nObservationsTable,
                                                      nObservationsBD,
                                                      nObservations);

    if (method == sumDense || method == sumCSR)
    {
        NumericTable *userSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
        BlockDescriptor<algorithmFPType> userSumsBD;
        userSumsTable->getBlockOfRows(0, 1, readOnly, userSumsBD);
        algorithmFPType *userSums = userSumsBD.getBlockPtr();

        daal_memcpy_s(*sums, nFeatures * sizeof(algorithmFPType), userSums, nFeatures * sizeof(algorithmFPType));

        userSumsTable->releaseBlockOfRows(userSumsBD);
    }
}

/*************************** releaseDenseCrossProductAndSums *************************************/
template<typename algorithmFPType, CpuType cpu>
void releaseDenseCrossProductAndSums( NumericTable                     *covTable,
                                      BlockDescriptor<algorithmFPType> &crossProductBD,
                                      NumericTable                     *meanTable,
                                      BlockDescriptor<algorithmFPType> &sumBD,
                                      NumericTable                     *nObservationsTable,
                                      BlockDescriptor<algorithmFPType> &nObservationsBD)
{
    covTable->releaseBlockOfRows(crossProductBD);
    meanTable->releaseBlockOfRows(sumBD);
    nObservationsTable->releaseBlockOfRows(nObservationsBD);
}


/********************* tls_data_t class *******************************************************/
template<typename algorithmFPType, CpuType cpu> struct tls_data_t
{
    algorithmFPType* crossProduct;
    algorithmFPType* sums;
    int malloc_errors;
    int isnormalized;

    tls_data_t(bool isNormalized, size_t nFeatures)
    {
        malloc_errors = 0;
        isnormalized = isNormalized;

        crossProduct  = service_scalable_calloc<algorithmFPType, cpu>(nFeatures * nFeatures);
        if(!crossProduct) { malloc_errors++; }

        if(!isnormalized)
        {
            sums  = service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
            if(!crossProduct) { malloc_errors++; }
        }
    }

    ~tls_data_t()
    {
        if(crossProduct) { service_scalable_free<algorithmFPType,cpu>( crossProduct ); crossProduct = 0; }
        if(!isnormalized)
        {
            if(sums) { service_scalable_free<algorithmFPType,cpu>( sums ); sums = 0; }
        }
    }
};


/********************* updateDenseCrossProductAndSums ********************************************/
template<typename algorithmFPType, Method method, CpuType cpu>
void updateDenseCrossProductAndSums(bool            isNormalized,
                                    bool            isOnline,
                                    size_t          nFeatures,
                                    size_t          nVectors,
                                    algorithmFPType *dataBlock,
                                    algorithmFPType *crossProduct,
                                    algorithmFPType *sums,
                                    algorithmFPType *nObservations,
                                    services::KernelErrorCollection *_errors)
{

    if((!isOnline) && ((isNormalized)  || ((!isNormalized) && (method == defaultDense))))
    {
        /* Inverse number of rows (for normalization) */
        algorithmFPType nVectorsInv = 1.0 / (double)(nVectors);

        /* Split rows by blocks */
        size_t numRowsInBlock = 140;
        size_t numBlocks = nVectors / numRowsInBlock;
        if (numBlocks * numRowsInBlock < nVectors) { numBlocks++; }

        /* TLS data initialization */
        daal::tls<tls_data_t<algorithmFPType, cpu> *> tls_data([ = ]()
        {
            return new tls_data_t<algorithmFPType, cpu>( isNormalized, nFeatures );
        });

        /* Threaded loop with syrk seq calls */
        daal::threader_for( numBlocks, numBlocks, [ & ](int iBlock)
        {
            struct tls_data_t<algorithmFPType,cpu> * tls_data_local = tls_data.local();
            if(tls_data_local->malloc_errors) return;

            char uplo  = 'U';
            char trans = 'N';
            algorithmFPType alpha = 1.0;
            algorithmFPType beta  = 1.0;


            size_t startRow = iBlock * numRowsInBlock;
            size_t endRow = startRow + numRowsInBlock;
            if (endRow > nVectors) { endRow = nVectors; }
            MKL_INT nFeatures_local = nFeatures;
            MKL_INT nVectors_local  = endRow - startRow;
            algorithmFPType* dataBlock_local = dataBlock + startRow * nFeatures;
            algorithmFPType* crossProduct_local =  tls_data_local->crossProduct;
            algorithmFPType* sums_local =  tls_data_local->sums;

            Blas<algorithmFPType, cpu>::xxsyrk( &uplo,
                                                &trans,
                                                (MKL_INT *) &nFeatures_local,
                                                (MKL_INT *) &nVectors_local,
                                                &alpha,
                                                dataBlock_local,
                                                (MKL_INT *) &nFeatures_local,
                                                &beta,
                                                crossProduct_local,
                                                (MKL_INT *) &nFeatures_local);

            if(!isNormalized)
            {
                /* Sum input array elements in case of non-normalized data */
                for( int i = 0; i < nVectors_local; i++)
                {
                   PRAGMA_IVDEP
                   PRAGMA_VECTOR_ALWAYS
                    for( int j = 0; j < nFeatures_local; j++)
                    {
                        sums_local[j] += dataBlock_local[i*nFeatures_local + j];
                    }
                }
            }
        } );

        /* TLS reduction: sum all partial cross products and sums */
        tls_data.reduce( [ = ]( tls_data_t<algorithmFPType,cpu>* tls_data_local )
        {
            if(tls_data_local->malloc_errors)
            {
                _errors->add(daal::services::ErrorMemoryAllocationFailed);
            }

            /* Sum all cross products */
            if(tls_data_local->crossProduct)
            {
               PRAGMA_IVDEP
               PRAGMA_VECTOR_ALWAYS
                for( size_t i = 0; i < (nFeatures * nFeatures); i++)
                {
                    crossProduct[i] += tls_data_local->crossProduct[i];
                }
            }

            /* Update sums vector in case of non-normalized data */
            if(!isNormalized)
            {
                if(tls_data_local->sums)
                {
                   PRAGMA_IVDEP
                   PRAGMA_VECTOR_ALWAYS
                    for( int i = 0; i < nFeatures; i++)
                    {
                        sums[i] += tls_data_local->sums[i];
                    }
                }
            }

            delete tls_data_local;
        } );

        /* If data is not normalized, perform subtractions of(sums[i]*sums[j])/n */
        if(!isNormalized)
        {
            for( int i = 0; i < nFeatures; i++ )
            {
               PRAGMA_IVDEP
               PRAGMA_VECTOR_ALWAYS
                for(int j = 0; j < nFeatures; j++ )
                {
                    crossProduct [i*nFeatures + j] -= (nVectorsInv * sums[i] * sums[j]);
                }
            }
        }
    }
    else
    {

        __int64 mklMethod = __DAAL_VSL_SS_METHOD_FAST;
        switch (method)
        {
        case defaultDense:
            mklMethod = __DAAL_VSL_SS_METHOD_FAST;
            break;
        case singlePassDense:
            mklMethod = __DAAL_VSL_SS_METHOD_1PASS;
            break;
        case sumDense:
            mklMethod = __DAAL_VSL_SS_METHOD_FAST_USER_MEAN;
            break;
        default:
            break;
        }

        int errcode = Statistics<algorithmFPType, cpu>::xcp(dataBlock, (__int64)nFeatures, (__int64)nVectors,
                                       nObservations, sums, crossProduct, mklMethod);
        if (errcode != 0) { _errors->add(services::ErrorCovarianceInternal); return; }
    }

    *nObservations += (algorithmFPType)nVectors;
}

/****************************** updateDensePartialResults ****************************************/
template<typename algorithmFPType, Method method, CpuType cpu>
void updateDensePartialResults( NumericTable *dataTable,
                                NumericTable *crossProductTable,
                                NumericTable *sumTable,
                                NumericTable *nObservationsTable,
                                bool         isOnline,
                                services::KernelErrorCollection *_errors)
{
    size_t nFeatures = dataTable->getNumberOfColumns();
    size_t nVectors  = dataTable->getNumberOfRows();
    bool isNormalized = dataTable->isNormalized(NumericTableIface::standardScoreNormalized);

    BlockDescriptor<algorithmFPType> crossProductBD, sumBD, nObservationsBD;
    algorithmFPType *crossProduct, *sums, *nObservations;
    ReadWriteMode rwMode = (isOnline ? readWrite : writeOnly);

    getDenseCrossProductAndSums<algorithmFPType, method, cpu>( nFeatures,
                                                               rwMode,
                                                               crossProductTable,
                                                               crossProductBD,
                                                               &crossProduct,
                                                               sumTable,
                                                               sumBD,
                                                               &sums,
                                                               nObservationsTable,
                                                               nObservationsBD,
                                                               &nObservations,
                                                               dataTable);

    if (!isOnline)
    {
        algorithmFPType zero = 0.0;
        daal::services::internal::service_memset<algorithmFPType, cpu>(crossProduct, zero, nFeatures * nFeatures);
        if (method != sumDense && method != sumCSR)
        {
            daal::services::internal::service_memset<algorithmFPType, cpu>(sums, zero, nFeatures);
        }
    }

    /* Retrieve data associated with input table */
    BlockDescriptor<algorithmFPType> dataBD;
    dataTable->getBlockOfRows(0, nVectors, readOnly, dataBD);
    algorithmFPType *dataBlock = dataBD.getBlockPtr();

    updateDenseCrossProductAndSums<algorithmFPType, method, cpu>( isNormalized,
                                                                  isOnline,
                                                                  nFeatures,
                                                                  nVectors,
                                                                  dataBlock,
                                                                  crossProduct,
                                                                  sums,
                                                                  nObservations,
                                                                  _errors);

    dataTable->releaseBlockOfRows(dataBD);

    releaseDenseCrossProductAndSums<algorithmFPType, cpu>( crossProductTable,
                                                           crossProductBD,
                                                           sumTable,
                                                           sumBD,
                                                           nObservationsTable,
                                                           nObservationsBD);
}

/********************** updateCSRCrossProductAndSums *********************************************/
template<typename algorithmFPType, Method method, CpuType cpu>
void updateCSRCrossProductAndSums( size_t           nFeatures,
                                   size_t           nVectors,
                                   algorithmFPType  *dataBlock,
                                   size_t           *colIndices,
                                   size_t           *rowOffsets,
                                   algorithmFPType  *crossProduct,
                                   algorithmFPType  *sums,
                                   algorithmFPType  *nObservations,
                                   services::KernelErrorCollection *_errors)
{
    char transa = 'T';
    SpBlas<algorithmFPType, cpu>::xcsrmultd( &transa,
                                            (MKL_INT *)&nVectors,
                                            (MKL_INT *)&nFeatures,
                                            (MKL_INT *)&nFeatures,
                                            dataBlock,
                                            (MKL_INT *)colIndices,
                                            (MKL_INT *)rowOffsets,
                                            dataBlock,
                                            (MKL_INT *)colIndices,
                                            (MKL_INT *)rowOffsets,
                                            crossProduct,
                                            (MKL_INT *)&nFeatures);

    if (method != sumCSR)
    {
        algorithmFPType one = 1.0;
        algorithmFPType *ones = (algorithmFPType *)daal_malloc(nVectors * sizeof(algorithmFPType));
        if (!ones) { _errors->add(services::ErrorMemoryAllocationFailed); return; }
        daal::services::internal::service_memset<algorithmFPType, cpu>(ones, one, nVectors);

        char matdescra[6];
        matdescra[0] = 'G';        // general matrix
        matdescra[3] = 'F';        // 1-based indexing

        matdescra[1] = (char) 0;
        matdescra[2] = (char) 0;
        matdescra[4] = (char) 0;
        matdescra[5] = (char) 0;
        SpBlas<algorithmFPType, cpu>::xcsrmv( &transa,
                                              (MKL_INT *)&nVectors,
                                              (MKL_INT *)&nFeatures,
                                              &one,
                                              matdescra,
                                              dataBlock,
                                              (MKL_INT *)colIndices,
                                              (MKL_INT *)rowOffsets,
                                              (MKL_INT *)rowOffsets + 1,
                                              ones,
                                              &one,
                                              sums);
        daal_free(ones);
    }

    nObservations[0] += (algorithmFPType)nVectors;
}

/*********************** mergeCrossProductAndSums ************************************************/
template<typename algorithmFPType, CpuType cpu>
void mergeCrossProductAndSums( size_t nFeatures,
                               const algorithmFPType *partialCrossProduct,
                               const algorithmFPType *partialSums,
                               const algorithmFPType *partialNObservations,
                               algorithmFPType *crossProduct,
                               algorithmFPType *sums,
                               algorithmFPType *nObservations)
{
    /* Merge cross-products */
    algorithmFPType partialNObsValue = partialNObservations[0];

    if (partialNObsValue != 0)
    {
        algorithmFPType nObsValue = nObservations[0];

        if (nObsValue == 0)
        {
            daal::threader_for( nFeatures, nFeatures, [ = ](size_t i)
            {
              PRAGMA_IVDEP
              PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
                    crossProduct[j * nFeatures + i]  = crossProduct[i * nFeatures + j];
                }
            } );
        }
        else
        {
            algorithmFPType invPartialNObs = 1.0 / partialNObsValue;
            algorithmFPType invNObs = 1.0 / nObsValue;
            algorithmFPType invNewNObs = 1.0 / (nObsValue + partialNObsValue);

            daal::threader_for( nFeatures, nFeatures, [ = ](size_t i)
            {
              PRAGMA_IVDEP
              PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
                    crossProduct[i * nFeatures + j] += partialSums[i] * partialSums[j] * invPartialNObs;
                    crossProduct[i * nFeatures + j] += sums[i] * sums[j] * invNObs;
                    crossProduct[i * nFeatures + j] -= (partialSums[i] + sums[i]) * (partialSums[j] + sums[j]) * invNewNObs;
                    crossProduct[j * nFeatures + i]  = crossProduct[i * nFeatures + j];
                }
            } );
        }

        /* Merge number of observations */
        nObservations[0] += partialNObservations[0];

        /* Merge sums */
        for (size_t i = 0; i < nFeatures; i++)
        {
            sums[i] += partialSums[i];
        }
    }
}

/*********************** finalizeCovariance ******************************************************/
template<typename algorithmFPType, CpuType cpu>
void finalizeCovariance( size_t          nFeatures,
                         algorithmFPType nObservations,
                         algorithmFPType *crossProduct,
                         algorithmFPType *sums,
                         algorithmFPType *cov,
                         algorithmFPType *mean,
                         const Parameter *parameter,
                         services::KernelErrorCollection *_errors)
{
    algorithmFPType invNObservations = 1.0 / nObservations;
    algorithmFPType invNObservationsM1 = 1.0;
    if (nObservations > 1.0)
    {
        invNObservationsM1 = 1.0 / (nObservations - 1.0);
    }

    /* Calculate resulting mean vector */
    for (size_t i = 0; i < nFeatures; i++)
    {
        mean[i] = sums[i] * invNObservations;
    }

    if (parameter->outputMatrixType == correlationMatrix)
    {
        /* Calculate resulting correlation matrix */
        algorithmFPType *diagInvSqrts = (algorithmFPType *)daal::services::daal_malloc(nFeatures * sizeof(algorithmFPType));
        if (!diagInvSqrts)
        { _errors->add(services::ErrorMemoryAllocationFailed); return; }

        for (size_t i = 0; i < nFeatures; i++)
        {
            diagInvSqrts[i] = 1.0 / daal::internal::Math<algorithmFPType,cpu>::sSqrt(crossProduct[i * nFeatures + i]);
        }

        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * diagInvSqrts[i] * diagInvSqrts[j];
            }
            cov[i * nFeatures + i] = 1.0; //diagonal element
        }

        daal::services::daal_free(diagInvSqrts);
    }
    else
    {
        /* Calculate resulting covariance matrix */
        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * invNObservationsM1;
            }
        }
    }

    /* Copy results into symmetric upper triangle */
    for (size_t i = 0; i < nFeatures; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            cov[j * nFeatures + i] = cov[i * nFeatures + j];
        }
    }
}

/*********************** finalizeCovariance ******************************************************/
template<typename algorithmFPType, CpuType cpu>
void finalizeCovariance( NumericTable *covTable,
                         NumericTable *meanTable,
                         NumericTable *nObservationsTable,
                         const Parameter         *parameter,
                         services::KernelErrorCollection *_errors)
{
    BlockDescriptor<algorithmFPType> covBD, meanBD, nObservationsBD;
    algorithmFPType *cov, *mean, *nObservations;

    getTableData<algorithmFPType, cpu>(readWrite, covTable,           covBD,           &cov);
    getTableData<algorithmFPType, cpu>(readWrite, meanTable,          meanBD,          &mean);
    getTableData<algorithmFPType, cpu>(readOnly,  nObservationsTable, nObservationsBD, &nObservations);

    size_t nFeatures = covTable->getNumberOfColumns();

    finalizeCovariance<algorithmFPType, cpu>( nFeatures,
                                              *nObservations,
                                              cov,
                                              mean,
                                              cov,
                                              mean,
                                              parameter,
                                              _errors);

    covTable->releaseBlockOfRows(covBD);
    meanTable->releaseBlockOfRows(meanBD);
    nObservationsTable->releaseBlockOfRows(nObservationsBD);
}

/************************ finalizeCovariance *******************************************************/
template<typename algorithmFPType, CpuType cpu>
void finalizeCovariance( NumericTable *crossProductTable,
                         NumericTable *sumTable,
                         NumericTable *nObservationsTable,
                         NumericTable *covTable,
                         NumericTable *meanTable,
                         const Parameter         *parameter,
                         services::KernelErrorCollection *_errors)
{
    size_t nFeatures = covTable->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> crossProductBD, sumBD, nObservationsBD;
    algorithmFPType *crossProduct, *sums, *nObservations;

    getDenseCrossProductAndSums<algorithmFPType, cpu>( readOnly,
                                                       crossProductTable,
                                                       crossProductBD,
                                                       &crossProduct,
                                                       sumTable,
                                                       sumBD,
                                                       &sums,
                                                       nObservationsTable,
                                                       nObservationsBD,
                                                       &nObservations);

    BlockDescriptor<algorithmFPType> covBD, meanBD;
    algorithmFPType *cov, *mean;
    getTableData<algorithmFPType, cpu>(writeOnly, covTable,  covBD,  &cov);
    getTableData<algorithmFPType, cpu>(writeOnly, meanTable, meanBD, &mean);

    finalizeCovariance<algorithmFPType, cpu>( nFeatures,
                                              *nObservations,
                                              crossProduct,
                                              sums,
                                              cov,
                                              mean,
                                              parameter,
                                              _errors);

    releaseDenseCrossProductAndSums<algorithmFPType, cpu>( crossProductTable,
                                                           crossProductBD,
                                                           sumTable,
                                                           sumBD,
                                                           nObservationsTable,
                                                           nObservationsBD);

    covTable->releaseBlockOfRows(covBD);
    meanTable->releaseBlockOfRows(meanBD);
}

} // namespace internal
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
