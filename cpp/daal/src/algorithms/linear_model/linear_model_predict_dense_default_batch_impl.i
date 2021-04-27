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
void PredictKernel<algorithmFPType, defaultDense, cpu>::computeBlockOfResponses(DAAL_INT * numFeatures, DAAL_INT * numRows,
                                                                                const algorithmFPType * dataBlock, DAAL_INT * numBetas,
                                                                                const algorithmFPType * beta, DAAL_INT * numResponses,
                                                                                algorithmFPType * responseBlock, bool findBeta0)
{
    /* GEMM parameters */
    char trans           = 'T';
    char notrans         = 'N';
    algorithmFPType one  = 1.0;
    algorithmFPType zero = 0.0;

    Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, numResponses, numRows, numFeatures, &one, beta + 1, numBetas, dataBlock, numFeatures, &zero,
                                       responseBlock, numResponses);

    if (findBeta0)
    {
        /* Add intercept term to linear regression results */
        DAAL_INT iZero             = 0;
        DAAL_INT numBetasValue     = *numBetas;
        DAAL_INT numResponsesValue = *numResponses;
        for (size_t j = 0; j < numResponsesValue; ++j)
        {
            Blas<algorithmFPType, cpu>::xxaxpy(numRows, &one, beta + j * numBetasValue, &iZero, responseBlock + j,
                                               numResponses);
        }
    } /* if (findBeta0) */
} /* void PredictKernel<algorithmFPType, defaultDense, cpu>::computeBlockOfResponses */

template <typename algorithmFPType, CpuType cpu>
void PredictKernel<algorithmFPType, defaultDense, cpu>::computeBlockOfResponsesSOA(const size_t & startRow, DAAL_INT * numFeatures,
                                                                                   DAAL_INT * numRows, NumericTable * dataTable, DAAL_INT * numBetas,
                                                                                   const algorithmFPType * beta, DAAL_INT * numResponses,
                                                                                   algorithmFPType * responseBlock, bool findBeta0)
{
    SafeStatus safeStat;
    algorithmFPType one       = 1.0;
    const DAAL_INT nFeatures  = *numFeatures;
    const DAAL_INT nRows      = *numRows;
    const DAAL_INT nResponses = *numResponses;
    const DAAL_INT numRowsInData = dataTable->getNumberOfRows();
    char trans                = 'T';
    char notrans              = 'N';

    services::internal::service_memset_seq<algorithmFPType, cpu>(responseBlock, algorithmFPType(0.0), nRows * nResponses);

    size_t numBlocks = nFeatures / blockSizeColumns + !!(nFeatures % blockSizeColumns);
    for (size_t iBlock = 0; iBlock < numBlocks; ++iBlock)
    {
        const size_t startCol = iBlock * blockSizeColumns;
        const DAAL_INT colSize   = (iBlock == numBlocks - 1) ? nFeatures - startCol : blockSizeColumns;

        ReadColumns<algorithmFPType, cpu> xBlock(dataTable, startCol, startRow, nRows);
        DAAL_CHECK_BLOCK_STATUS_THR(xBlock);
        const algorithmFPType * data = xBlock.get();
        Blas<algorithmFPType, cpu>::xxgemm(&trans, &trans, numResponses, numRows, &colSize, &one, beta + 1 + startCol, numBetas, data, &numRowsInData,
                                           &one, responseBlock, numResponses);
    }
    if (findBeta0)
    {
        DAAL_INT zero        = 0;
        const size_t nBetas = *numBetas;
        for (size_t j = 0; j < nResponses; ++j)
        {
            Blas<algorithmFPType, cpu>::xxaxpy(numRows, &one, beta + j * nBetas, &zero, responseBlock + j,
                                               numResponses);
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictKernel<algorithmFPType, defaultDense, cpu>::compute(const NumericTable * a, const linear_model::Model * m, NumericTable * r)
{
    linear_model::Model * model = const_cast<linear_model::Model *>(m);

    /* Get numeric tables with input data */
    NumericTable * dataTable = const_cast<NumericTable *>(a);

    /* Get numeric table to store results */
    DAAL_INT numVectors       = dataTable->getNumberOfRows();
    DAAL_INT numColumnsInData = dataTable->getNumberOfColumns();

    /* Get linear regression coefficients */
    NumericTable * betaTable = model->getBeta().get();
    DAAL_INT numResponses    = betaTable->getNumberOfRows();

    /* Retrieve data associated with coefficients */
    ReadRows<algorithmFPType, cpu> betaRows(betaTable, 0, numResponses);
    DAAL_CHECK_BLOCK_STATUS(betaRows)
    const algorithmFPType * beta = betaRows.get();

    size_t numRowsInBlock = blockSizeRows;

    /* Calculate number of blocks of rows including tail block */
    size_t numBlocks = numVectors / numRowsInBlock + !!(numVectors / numRowsInBlock);

    SafeStatus safeStat;
    daal::threader_for(numBlocks, numBlocks, [=, &safeStat](int iBlock) {
        size_t startRow  = iBlock * numRowsInBlock;
        size_t numRows = (iBlock == numBlocks - 1) ? numVectors - startRow : numRowsInBlock;
        
        DAAL_INT numFeatures = dataTable->getNumberOfColumns();
        DAAL_INT nAllBetas   = betaTable->getNumberOfColumns();

        Status s;

        WriteOnlyRows<algorithmFPType, cpu> responseRows(r, startRow, numRows);
        s = responseRows.status();
        if (!s)
        {
            safeStat |= s;
            return;
        }
        algorithmFPType * responseBlock = responseRows.get();
        DAAL_INT * pnumResponses   = (DAAL_INT *)&numResponses;
        DAAL_INT * pnumRows        = (DAAL_INT *)&numRows;

        if (dataTable->getDataLayout() & NumericTableIface::soa)
        {
            /* Calculate predictions */
            computeBlockOfResponsesSOA(startRow, &numFeatures, pnumRows, dataTable, &nAllBetas, beta, pnumResponses, responseBlock,
                                       model->getInterceptFlag());
        } else {
            /* Retrieve data blocks associated with input and resulting tables */
            ReadRows<algorithmFPType, cpu> dataRows(dataTable, startRow, numRows);
            s = dataRows.status();
            if (!s)
            {
                safeStat |= s;
                return;
            }
            const algorithmFPType * dataBlock = dataRows.get();
            /* Calculate predictions */
            computeBlockOfResponses(&numFeatures, pnumRows, dataBlock, &nAllBetas, beta, pnumResponses, responseBlock, model->getInterceptFlag());
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
