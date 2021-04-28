/* file: linear_model_predict_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Common functions for linear regression predictions calculation
//--
*/

#ifndef __LINEAR_MODEL_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __LINEAR_MODEL_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/services/service_defines.h"
#include "src/externals/service_blas.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;

/**
 *  \brief Function that computes linear regression prediction results
 *         for a block of input data rows
 *
 *  \param numFeatures[in]      Number of features in input data row
 *  \param numRows[in]          Number of input data rows
 *  \param dataBlock[in]        Block of input data rows
 *  \param numBetas[in]         Number of regression coefficients
 *  \param beta[in]             Regression coefficients
 *  \param numResponses[in]     Number of responses to calculate for each input data row
 *  \param responseBlock[out]   Resulting block of responses
 *  \param findBeta0[in]        Flag. True if regression coefficient contain intercept term;
 *                              false - otherwise.
 */
template <typename algorithmFPType, CpuType cpu>
services::Status PredictKernel<algorithmFPType, defaultDense, cpu>::computeBlockOfResponses(size_t startRow, size_t numFeatures, size_t numRows,
                                                                                            NumericTable * dataTable, size_t numBetas,
                                                                                            const algorithmFPType * beta, size_t numResponses,
                                                                                            algorithmFPType * responseBlock, bool findBeta0)
{
    Status st;
    /* Retrieve data blocks associated with input and resulting tables */
    ReadRows<algorithmFPType, cpu> dataRows(dataTable, startRow, numRows);
    DAAL_CHECK_BLOCK_STATUS(dataRows);
    const algorithmFPType * dataBlock = dataRows.get();

    /* GEMM parameters */
    char trans           = 'T';
    char notrans         = 'N';
    algorithmFPType one  = 1.0;
    algorithmFPType zero = 0.0;

    Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, (DAAL_INT *)&numResponses, (DAAL_INT *)&numRows, (DAAL_INT *)&numFeatures, &one, beta + 1,
                                       (DAAL_INT *)&numBetas, dataBlock, (DAAL_INT *)&numFeatures, &zero, responseBlock, (DAAL_INT *)&numResponses);

    if (findBeta0)
    {
        /* Add intercept term to linear regression results */
        DAAL_INT iZero = 0;
        for (size_t j = 0; j < numResponses; ++j)
        {
            Blas<algorithmFPType, cpu>::xxaxpy((DAAL_INT *)&numRows, &one, beta + j * numBetas, &iZero, responseBlock + j, (DAAL_INT *)&numResponses);
        }
    } /* if (findBeta0) */
    return st;
} /* void PredictKernel<algorithmFPType, defaultDense, cpu>::computeBlockOfResponses */

template <typename algorithmFPType, CpuType cpu>
services::Status PredictKernel<algorithmFPType, defaultDense, cpu>::computeBlockOfResponsesSOA(size_t startRow, size_t numFeatures,
                                                                                               size_t numRowsInBlock, NumericTable * dataTable,
                                                                                               size_t numBetas, const algorithmFPType * beta,
                                                                                               size_t numResponses, algorithmFPType * responseBlock,
                                                                                               bool findBeta0, TlsMem<algorithmFPType, cpu> & tlsData)
{
    Status st;
    char trans                       = 'T';
    char notrans                     = 'N';
    algorithmFPType one              = 1.0;
    const size_t nRowsInData         = dataTable->getNumberOfRows();
    const size_t numBlockSizeColumns = blockSizeColumns;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, numRowsInBlock, numResponses);
    services::internal::service_memset_seq<algorithmFPType, cpu>(responseBlock, algorithmFPType(0.0), numRowsInBlock * numResponses);

    const size_t numBlocks = numFeatures / blockSizeColumns + !!(numFeatures % blockSizeColumns);
    for (size_t iBlock = 0; iBlock < numBlocks; ++iBlock)
    {
        const size_t startColumn       = iBlock * blockSizeColumns;
        const size_t numColumnsInBlock = (iBlock == numBlocks - 1) ? numFeatures - startColumn : blockSizeColumns;

        if (static_cast<const SOANumericTable &>(*dataTable).isHomogeneousFloatOrDouble())
        {
            ReadColumns<algorithmFPType, cpu> xBlock(dataTable, startColumn, startRow, numRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS(xBlock);
            const algorithmFPType * data = xBlock.get();
            Blas<algorithmFPType, cpu>::xxgemm(&trans, &trans, (DAAL_INT *)&numResponses, (DAAL_INT *)&numRowsInBlock, (DAAL_INT *)&numColumnsInBlock,
                                               &one, beta + 1 + startColumn, (DAAL_INT *)&numBetas, data, (DAAL_INT *)&nRowsInData, &one,
                                               responseBlock, (DAAL_INT *)&numResponses);
        }
        else
        {
            algorithmFPType * tlsLocal = tlsData.local();
            DAAL_CHECK_MALLOC(tlsLocal);
            for (size_t i = 0; i < numColumnsInBlock; ++i)
            {
                ReadColumns<algorithmFPType, cpu> xBlock(dataTable, i + startColumn, startRow, numRowsInBlock);
                DAAL_CHECK_BLOCK_STATUS(xBlock);
                services::internal::tmemcpy<algorithmFPType, cpu>(tlsLocal + i * blockSizeRows, xBlock.get(), numRowsInBlock);
            }
            Blas<algorithmFPType, cpu>::xxgemm(&trans, &trans, (DAAL_INT *)&numResponses, (DAAL_INT *)&numRowsInBlock, (DAAL_INT *)&numColumnsInBlock,
                                               &one, beta + 1 + startColumn, (DAAL_INT *)&numBetas, tlsLocal, (DAAL_INT *)&numBlockSizeColumns, &one,
                                               responseBlock, (DAAL_INT *)&numResponses);
        }
    }
    if (findBeta0)
    {
        const DAAL_INT zero = 0;
        for (size_t j = 0; j < numResponses; ++j)
        {
            Blas<algorithmFPType, cpu>::xxaxpy((DAAL_INT *)&numRowsInBlock, &one, beta + j * numBetas, &zero, responseBlock + j,
                                               (DAAL_INT *)&numResponses);
        }
    }
    return st;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictKernel<algorithmFPType, defaultDense, cpu>::compute(const NumericTable * a, const linear_model::Model * m, NumericTable * r)
{
    linear_model::Model * model = const_cast<linear_model::Model *>(m);

    /* Get numeric tables with input data */
    NumericTable * dataTable = const_cast<NumericTable *>(a);

    /* Get sizes of input data */
    const size_t numVectors  = dataTable->getNumberOfRows();
    const size_t numFeatures = dataTable->getNumberOfColumns();

    /* Get linear regression coefficients */
    NumericTable * betaTable  = model->getBeta().get();
    const size_t numResponses = betaTable->getNumberOfRows();
    const size_t nAllBetas    = betaTable->getNumberOfColumns();

    /* Retrieve data associated with coefficients */
    ReadRows<algorithmFPType, cpu> betaRows(betaTable, 0, numResponses);
    DAAL_CHECK_BLOCK_STATUS(betaRows)
    const algorithmFPType * beta = betaRows.get();

    /* Calculate number of blocks */
    const size_t numBlocks = numVectors / blockSizeRows + !!(numVectors % blockSizeRows);

    SafeStatus safeStat;
    TlsMem<algorithmFPType, cpu> tlsData(blockSizeRows * blockSizeColumns);

    daal::threader_for(numBlocks, numBlocks, [&](int iBlock) {
        const size_t startRow       = iBlock * blockSizeRows;
        const size_t numRowsInBlock = (iBlock == numBlocks - 1) ? numVectors - startRow : blockSizeRows;

        Status s;

        WriteOnlyRows<algorithmFPType, cpu> responseRows(r, startRow, numRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(responseRows);
        algorithmFPType * responseBlock = responseRows.get();

        /* Calculate predictions */
        if (dataTable->getDataLayout() & NumericTableIface::soa)
        {
            DAAL_CHECK_STATUS_THR(computeBlockOfResponsesSOA(startRow, numFeatures, numRowsInBlock, dataTable, nAllBetas, beta, numResponses,
                                                             responseBlock, model->getInterceptFlag(), tlsData));
        }
        else
        {
            DAAL_CHECK_STATUS_THR(computeBlockOfResponses(startRow, numFeatures, numRowsInBlock, dataTable, nAllBetas, beta, numResponses,
                                                          responseBlock, model->getInterceptFlag()));
        }
    }); /* daal::threader_for */
    return safeStat.detach();
} /* void PredictKernel<algorithmFPType, defaultDense, cpu>::compute */

} /* namespace internal */
} /* namespace prediction */
} /* namespace linear_model */
} /* namespace algorithms */
} /* namespace daal */

#endif
