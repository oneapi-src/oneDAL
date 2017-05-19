/* file: linear_regression_train_dense_normeq_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of auxiliary functions for linear regression
//  Normal Equations (normEqDense) method.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_lapack.h"
#include "linear_regression_ne_model.h"
#include "linear_regression_train_kernel.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_memory.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{

/**
 *  \brief Get arrays holding partial sums from Linear Regression daal::algorithms::Model
 *
 *  \param  daal::algorithms::Model[in]     Linear regression daal::algorithms::Model
 *  \param  dim[in]       Task dimension
 *  \param  ny[in]        Number of responses
 *  \param  rwmode[in]    Flag specifying read/write access to the daal::algorithms::Model's partial results
 *
 *  \param  xtxTable[out] Numeric table containing matrix X'*X
 *  \param  xtxBD[out]    Buffer manager corresponding to xtxTable
 *  \param  xtx[out]      Array containing matrix X'*X
 *
 *  \param  xtyTable[out] Numeric table containing matrix X'*Y
 *  \param  xtyBD[out]    Buffer manager corresponding to xtyTable
 *  \param  xty[out]      Array containing matrix X'*Y
 */
template<typename algorithmFPType, CpuType cpu>
static void getModelPartialSums(ModelNormEq *model,
                                DAAL_INT dim, DAAL_INT ny, ReadWriteMode rwmode,
                                NumericTable **xtxTable, BlockDescriptor<algorithmFPType> &xtxBD, algorithmFPType **xtx,
                                NumericTable **xtyTable, BlockDescriptor<algorithmFPType> &xtyBD, algorithmFPType **xty)
{
    *xtxTable = model->getXTXTable().get();
    *xtyTable = model->getXTYTable().get();

    (*xtxTable)->getBlockOfRows(0, dim, rwmode, xtxBD);
    *xtx = xtxBD.getBlockPtr();
    (*xtyTable)->getBlockOfRows(0, ny,  rwmode, xtyBD);
    *xty = xtyBD.getBlockPtr();
}

/**
 *  \brief Release arrays holding partial sums in Linear Regression daal::algorithms::Model
 *
 *  \param  xtxTable[in]  Numeric table containing matrix X'*X
 *  \param  xtxBD[in]     Buffer manager corresponding to xtxTable
 *  \param  xtyTable[in]  Numeric table containing matrix X'*Y
 *  \param  xtyBD[in]     Buffer manager corresponding to xtyTable
 */
template<typename algorithmFPType, CpuType cpu>
static void releaseModelNormEqPartialSums(NumericTable *xtxTable, BlockDescriptor<algorithmFPType> &xtxBD,
                                          NumericTable *xtyTable, BlockDescriptor<algorithmFPType> &xtyBD)
{
    xtxTable->releaseBlockOfRows(xtxBD);
    xtyTable->releaseBlockOfRows(xtyBD);
}



template <typename algorithmFPType, CpuType cpu>
static void updatePartialSums(
                       DAAL_INT *p,          /* features */
                       DAAL_INT *n,          /* vectors */
                       DAAL_INT *b,          /* features + 1 */
                       algorithmFPType  *x_in,       /* p*n input matrix */
                       algorithmFPType  *xtx_out,    /* p*b output matrix */
                       DAAL_INT *v,          /* variables */
                       algorithmFPType  *y_in,       /* v*n input matrix   */
                       algorithmFPType  *xty_out     /* v*b output matrix */
                      )
{
size_t i, j;

DAAL_INT p_val = *p;
DAAL_INT n_val = *n;
DAAL_INT b_val = *b;
DAAL_INT v_val = *v;

char uplo   = 'U';
char trans  = 'N';
char transa = 'N';
char transb = 'T';

algorithmFPType alpha = 1.0;
algorithmFPType *xtx_ptr;
algorithmFPType *x_ptr;
algorithmFPType *y_ptr;

    Blas<algorithmFPType, cpu>::xxsyrk(&uplo, &trans, p, n, &alpha, x_in, p, &alpha, xtx_out, b);

    if ( p_val < b_val )
    {
        xtx_ptr = xtx_out + p_val * b_val;

      PRAGMA_IVDEP
      PRAGMA_VECTOR_ALWAYS
        for ( i = 0, x_ptr = x_in; i < n_val; i++, x_ptr += p_val)
        {
            for ( j = 0; j < p_val; j++)
            {
                xtx_ptr[j] += x_ptr[j];
            }
        }

        xtx_ptr[p_val] += (algorithmFPType)n_val;

    } /* if ( p_val < b_val ) */

    Blas<algorithmFPType, cpu>::xxgemm(&transa, &transb, p, v, n, &alpha, x_in, p, y_in, v, &alpha, xty_out, b);

    if ( p_val < b_val )
    {
        for ( i = 0, y_ptr = y_in; i < n_val; i++, y_ptr += v_val)
        {
          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for (j = 0; j < v_val; j++)
            {
                xty_out[j * b_val + p_val] += y_ptr[j];
            }
        }

    } /* if ( p_val < b_val ) */

} /* updatePartialSums */

/**
 *  \brief Function that calculates linear regression coefficients
 *         from matrices X'*X and X'*Y.
 *
 *  \param p[in]        Number of rows in input matrix X'*X
 *  \param xtx[in]      Input matrix X'*X
 *  \param ldxtx[in]    Leading dimension of matrix X'*X (ldxtx >= p)
 *  \param ny[in]       Number of rows in input matrix X'*Y
 *  \param xty[in]      Input matrix X'*Y
 *  \param ldxty[in]    Leading dimension of matrix X'*Y (ldxty >= p)
 *  \param beta[out]    Resulting matrix of coefficients of size ny x ldxty
 */
template <typename algorithmFPType, CpuType cpu>
static void computeLinregCoeffs(DAAL_INT *p,  algorithmFPType *xtx, DAAL_INT *ldxtx,
                                DAAL_INT *ny, algorithmFPType *xty, DAAL_INT *ldxty, algorithmFPType *beta,
                                services::KernelErrorCollection *_errors)
{
    DAAL_INT n;
    DAAL_INT i_one = 1;
    char uplo = 'U';
    DAAL_INT info;

    n = (*ny) * (*ldxty);
    daal::services::daal_memcpy_s(beta, n * sizeof(algorithmFPType), xty, n * sizeof(algorithmFPType));

    const size_t xtxSizeInBytes = (*p) * (*ldxtx) * sizeof(algorithmFPType);
    algorithmFPType * const tempXTX = static_cast<algorithmFPType *>(daal::services::daal_malloc(xtxSizeInBytes));
    if (!tempXTX) { _errors->add(services::ErrorMemoryAllocationFailed); return; }
    daal::services::daal_memcpy_s(tempXTX, xtxSizeInBytes, xtx, xtxSizeInBytes);

    /* Perform L*L' decomposition of X'*X */
    Lapack<algorithmFPType, cpu>::xpotrf(&uplo, p, tempXTX, ldxtx, &info);
    if (info < 0) { daal::services::daal_free(tempXTX); _errors->add(services::ErrorLinearRegressionInternal); return; }
    if (info > 0) { daal::services::daal_free(tempXTX); _errors->add(services::ErrorNormEqSystemSolutionFailed); return; }

    /* Solve L*L'*b=Y */
    Lapack<algorithmFPType, cpu>::xpotrs(&uplo, p, ny, tempXTX, ldxtx, beta, ldxty, &info);
    if (info != 0) { daal::services::daal_free(tempXTX); _errors->add(services::ErrorLinearRegressionInternal); return; }

    daal::services::daal_free(tempXTX);
} /* computeLinregCoeffs */


template <typename algorithmFPType, CpuType cpu>
void updatePartialModelNormEq(NumericTable *x, NumericTable *y,
            linear_regression::Model *r,
            const daal::algorithms::Parameter *par, bool isOnline,
            services::KernelErrorCollection *_errors)
{
    const linear_regression::Parameter *parameter = static_cast<const linear_regression::Parameter *>(par);
    ModelNormEq *rr = static_cast<ModelNormEq *>(r);

    DAAL_INT nRows      = (DAAL_INT)x->getNumberOfRows();     /* vectors */
    DAAL_INT nFeatures  = (DAAL_INT)x->getNumberOfColumns();  /* features */
    DAAL_INT nResponses = (DAAL_INT)y->getNumberOfColumns();  /* variables */
    DAAL_INT nBetas     = (DAAL_INT)rr->getNumberOfBetas();   /* features + 1 */

    DAAL_INT nBetasIntercept = nBetas;
    if (parameter && !parameter->interceptFlag) { nBetasIntercept--; }; /* features + 1 */

    /* Retrieve matrices X'*X and X'*Y from daal::algorithms::Model */
    NumericTable *xtxTable, *xtyTable;
    BlockDescriptor<algorithmFPType> xtxBD, xtyBD;
    algorithmFPType *xtx, *xty;

    getModelPartialSums<algorithmFPType, cpu>(rr, nBetasIntercept, nResponses, readWrite, &xtxTable, xtxBD, &xtx, &xtyTable, xtyBD, &xty);

    /* Retrieve data associated with input tables */
    BlockDescriptor<algorithmFPType> xBD;
    BlockDescriptor<algorithmFPType> yBD;

    x->getBlockOfRows(0, nRows, readOnly, xBD);
    y->getBlockOfRows(0, nRows, readOnly, yBD);

    algorithmFPType *dx = xBD.getBlockPtr();
    algorithmFPType *dy = yBD.getBlockPtr();

    /* Initialize output arrays by zero in case of batch mode */
    if(!isOnline)
    {
        daal::services::internal::service_memset<algorithmFPType, cpu>(xtx, 0, nBetasIntercept * nBetasIntercept);
        daal::services::internal::service_memset<algorithmFPType, cpu>(xty, 0, nResponses * nBetasIntercept);
    }

    /* Split rows by blocks */
    size_t numRowsInBlock = 128;

    size_t numBlocks = nRows / numRowsInBlock;
    if (numBlocks * numRowsInBlock < nRows) { numBlocks++; }

    /* Create TLS xtx buffer */
    daal::tls<algorithmFPType *> xtx_buff( [ = ]()-> algorithmFPType*
    {
        algorithmFPType *ptr = service_scalable_calloc<algorithmFPType, cpu>(nBetasIntercept * nBetasIntercept);
        if (!ptr) { _errors->add(services::ErrorMemoryAllocationFailed); }
        return ptr;
    } );

    /* Create TLS xty buffer */
    daal::tls<algorithmFPType *> xty_buff( [ = ]()-> algorithmFPType*
    {
        algorithmFPType *ptr = service_scalable_calloc<algorithmFPType, cpu>(nResponses * nBetasIntercept);
        if (!ptr) { _errors->add(services::ErrorMemoryAllocationFailed); }
        return ptr;
    } );

    /* Intel(R) TBB threaded loop */
    daal::threader_for( numBlocks, numBlocks, [ =, &xtx_buff, &xty_buff ](int iBlock)
    {
        algorithmFPType *xtx_local =  xtx_buff.local();
        algorithmFPType *xty_local =  xty_buff.local();

        size_t startRow = iBlock * numRowsInBlock;
        size_t endRow = startRow + numRowsInBlock;
        if (endRow > nRows) { endRow = nRows; }

        algorithmFPType *dx_ptr = dx + startRow * nFeatures;
        algorithmFPType *dy_ptr = dy + startRow * nResponses;

        DAAL_INT nP = nFeatures;
        DAAL_INT nN = endRow - startRow;
        DAAL_INT nB = nBetasIntercept;
        DAAL_INT nV = nResponses;

        updatePartialSums<algorithmFPType, cpu>(&nP, &nN, &nB, dx_ptr, xtx_local, &nV, dy_ptr, xty_local);
    } );

    /* Sum all xtx and free buffer */
    xtx_buff.reduce( [ = ](algorithmFPType * v)-> void
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for( size_t i = 0; i < (nBetasIntercept * nBetasIntercept); i++) { xtx[i] += v[i]; }
        service_scalable_free<algorithmFPType, cpu>( v );
    } );

    /* Sum all xty and free buffer */
    xty_buff.reduce( [ = ](algorithmFPType * v)-> void
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for( size_t i = 0; i < (nResponses * nBetasIntercept); i++) { xty[i] += v[i]; }
        service_scalable_free<algorithmFPType, cpu>( v );
    } );

    x->releaseBlockOfRows(xBD);
    y->releaseBlockOfRows(yBD);

    releaseModelNormEqPartialSums<algorithmFPType, cpu>(xtxTable, xtxBD, xtyTable, xtyBD);

} /* updatePartialModelNormEq */

template<typename algorithmFPType, CpuType cpu>
static void copyModelIntermediateTable(size_t srcSize, const algorithmFPType * src, NumericTable & dest)
{
    BlockDescriptor<algorithmFPType> destBD;
    const size_t destNumberOfRows = dest.getNumberOfRows();
    dest.getBlockOfRows(0, destNumberOfRows, writeOnly, destBD);
    daal::services::daal_memcpy_s(destBD.getBlockPtr(), destNumberOfRows * dest.getNumberOfColumns() * sizeof(algorithmFPType),
                                  src, srcSize * sizeof(algorithmFPType));
    dest.releaseBlockOfRows(destBD);
}

template <typename algorithmFPType, CpuType cpu>
void finalizeModelNormEq(linear_regression::Model *a, linear_regression::Model *r,
                   services::KernelErrorCollection *_errors)
{
    ModelNormEq *aa = static_cast<ModelNormEq *>(a);
    ModelNormEq *rr = static_cast<ModelNormEq *>(r);

    DAAL_INT nBetas = (DAAL_INT)rr->getNumberOfBetas();
    DAAL_INT nResponses = (DAAL_INT)rr->getNumberOfResponses();
    DAAL_INT nBetasIntercept = nBetas;
    if (!rr->getInterceptFlag()) { nBetasIntercept--; }

    algorithmFPType *betaBuffer = (algorithmFPType *)daal::services::daal_malloc(nResponses * nBetas * sizeof(algorithmFPType));
    if (!betaBuffer) { _errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Retrieve matrices X'*X and X'*Y from daal::algorithms::Model */
    NumericTable *xtxTable, *xtyTable;
    BlockDescriptor<algorithmFPType> xtxBD, xtyBD;
    algorithmFPType *xtx, *xty;
    getModelPartialSums<algorithmFPType, cpu>(aa, nBetas, nResponses, readOnly, &xtxTable, xtxBD, &xtx, &xtyTable, xtyBD, &xty);

    if (aa != rr)
    {
        copyModelIntermediateTable<algorithmFPType, cpu>(xtxBD.getNumberOfRows() * xtxBD.getNumberOfColumns(), xtx, *(rr->getXTXTable()));
        copyModelIntermediateTable<algorithmFPType, cpu>(xtyBD.getNumberOfRows() * xtyBD.getNumberOfColumns(), xty, *(rr->getXTYTable()));
    }

    computeLinregCoeffs<algorithmFPType, cpu>(&nBetasIntercept, xtx, &nBetasIntercept, &nResponses, xty, &nBetasIntercept, betaBuffer, _errors);

    releaseModelNormEqPartialSums<algorithmFPType, cpu>(xtxTable, xtxBD, xtyTable, xtyBD);

    NumericTable *betaTable = rr->getBeta().get();
    BlockDescriptor<algorithmFPType> betaBD;
    betaTable->getBlockOfRows(0, nResponses, writeOnly, betaBD);
    algorithmFPType *beta = betaBD.getBlockPtr();

    if (nBetasIntercept == nBetas)
    {
        for(size_t i = 0; i < nResponses; i++)
        {
          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for(size_t j = 1; j < nBetas; j++)
            {
                beta[i * nBetas + j] = betaBuffer[i * nBetas + j - 1];
            }
            beta[i * nBetas] = betaBuffer[i * nBetas + nBetas - 1];
        }
    }
    else
    {
        for(size_t i = 0; i < nResponses; i++)
        {
          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for(size_t j = 0; j < nBetas - 1; j++)
            {
                beta[i * nBetas + j + 1] = betaBuffer[i * nBetasIntercept + j];
            }
            beta[i * nBetas] = 0.0;
        }
    }

    betaTable->releaseBlockOfRows(betaBD);
    daal::services::daal_free(betaBuffer);

} /* finalizeModelNormEq */


template <typename algorithmFPType, CpuType cpu>
services::Status LinearRegressionTrainBatchKernel<algorithmFPType, training::normEqDense, cpu>::compute(
    NumericTable *x, NumericTable *y, linear_regression::Model *r,
    const daal::algorithms::Parameter *par)
{
    bool isOnline = false;
    updatePartialModelNormEq<algorithmFPType, cpu>(x, y, r, par, isOnline, this->_errors.get());
    finalizeModelNormEq<algorithmFPType, cpu>(r, r, this->_errors.get());
    DAAL_RETURN_STATUS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status LinearRegressionTrainOnlineKernel<algorithmFPType, training::normEqDense, cpu>::compute(
    NumericTable *x, NumericTable *y, linear_regression::Model *r,
    const daal::algorithms::Parameter *par)
{
    bool isOnline = true;
    updatePartialModelNormEq<algorithmFPType, cpu>(x, y, r, par, isOnline, this->_errors.get());
    DAAL_RETURN_STATUS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status LinearRegressionTrainOnlineKernel<algorithmFPType, training::normEqDense, cpu>::finalizeCompute(
    linear_regression::Model *a, linear_regression::Model *r,
    const daal::algorithms::Parameter *par)
{
    finalizeModelNormEq<algorithmFPType, cpu>(a, r, this->_errors.get());
    DAAL_RETURN_STATUS();
}

} /* namespace internal */
} /* namespace training */
} /* namespace linear_regression */
} /* namespace algorithms */
} /* namespace daal */

#endif
