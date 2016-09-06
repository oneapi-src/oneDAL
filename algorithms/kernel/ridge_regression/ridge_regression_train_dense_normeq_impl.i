/* file: ridge_regression_train_dense_normeq_impl.i */
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
//  Implementation of auxiliary functions for ridge regression Normal Equations (normEqDense) method.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__
#define __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_lapack.h"
#include "ridge_regression_ne_model.h"
#include "ridge_regression_train_kernel.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_memory.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{

using namespace daal::services::internal;
using namespace daal::internal;

namespace training
{
namespace internal
{

/**
 *  \brief Get arrays holding partial sums from Ridge Regression daal::algorithms::Model
 *
 *  \param  daal::algorithms::Model[in]     Ridge regression daal::algorithms::Model
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
template<typename algorithmFpType, CpuType cpu>
static void getModelPartialSums(ModelNormEq *model, MKL_INT dim, MKL_INT ny, ReadWriteMode rwmode,
                                NumericTable **xtxTable, BlockDescriptor<algorithmFpType> &xtxBD, algorithmFpType **xtx,
                                NumericTable **xtyTable, BlockDescriptor<algorithmFpType> &xtyBD, algorithmFpType **xty)
{
    *xtxTable = model->getXTXTable().get();
    *xtyTable = model->getXTYTable().get();

    (*xtxTable)->getBlockOfRows(0, dim, rwmode, xtxBD);
    *xtx = xtxBD.getBlockPtr();
    (*xtyTable)->getBlockOfRows(0, ny,  rwmode, xtyBD);
    *xty = xtyBD.getBlockPtr();
}

/**
 *  \brief Release arrays holding partial sums in Ridge Regression daal::algorithms::Model
 *
 *  \param  xtxTable[in]  Numeric table containing matrix X'*X
 *  \param  xtxBD[in]     Buffer manager corresponding to xtxTable
 *  \param  xtyTable[in]  Numeric table containing matrix X'*Y
 *  \param  xtyBD[in]     Buffer manager corresponding to xtyTable
 */
template<typename algorithmFpType, CpuType cpu>
static void releaseModelNormEqPartialSums(NumericTable *xtxTable, BlockDescriptor<algorithmFpType> &xtxBD,
                                          NumericTable *xtyTable, BlockDescriptor<algorithmFpType> &xtyBD)
{
    xtxTable->releaseBlockOfRows(xtxBD);
    xtyTable->releaseBlockOfRows(xtyBD);
}

template <typename algorithmFpType, CpuType cpu>
static void updatePartialSums(MKL_INT *p,                   /* features */
                              MKL_INT *n,                   /* vectors */
                              MKL_INT *b,                   /* features + 1 */
                              algorithmFpType  *x_in,       /* p*n input matrix */
                              algorithmFpType  *xtx_out,    /* p*b output matrix */
                              MKL_INT *v,                   /* variables */
                              algorithmFpType  *y_in,       /* v*n input matrix   */
                              algorithmFpType  *xty_out     /* v*b output matrix */
                              )
{
    size_t i, j;

    MKL_INT p_val = *p;
    MKL_INT n_val = *n;
    MKL_INT b_val = *b;
    MKL_INT v_val = *v;

    char uplo   = 'U';
    char trans  = 'N';
    char transa = 'N';
    char transb = 'T';

    algorithmFpType alpha = 1.0;
    algorithmFpType* xtx_ptr;
    algorithmFpType* x_ptr;
    algorithmFpType* y_ptr;

    Blas<algorithmFpType, cpu>::xxsyrk(&uplo, &trans, p, n, &alpha, x_in, p, &alpha, xtx_out, b);

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

        xtx_ptr[p_val] += (algorithmFpType)n_val;

    } /* if ( p_val < b_val ) */

    Blas<algorithmFpType, cpu>::xxgemm(&transa, &transb, p, v, n, &alpha, x_in, p, y_in, v, &alpha, xty_out, b);

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

} // updatePartialSums

/**
 *  \brief Function that calculates ridge regression coefficients
 *         from matrices X'*X and X'*Y.
 *
 *  \param p[in]               Number of rows in input matrix X'*X
 *  \param xtx[in]             Input matrix X'*X
 *  \param ldxtx[in]           Leading dimension of matrix X'*X (ldxtx >= p)
 *  \param ny[in]              Number of rows in input matrix X'*Y
 *  \param xty[in]             Input matrix X'*Y
 *  \param ldxty[in]           Leading dimension of matrix X'*Y (ldxty >= p)
 *  \param isEqualRidge[in]    Flag that indicates if equal ridge parameters must be used
 *  \param ridge[in]           Ridge parameters of size 1 x 1 or 1 x ny (depends on isEqualRidge)
 *  \param beta[out]           Resulting matrix of coefficients of size ny x ldxty
 */
template <typename algorithmFpType, CpuType cpu>
static void computeRidgeCoeffs(MKL_INT *p, algorithmFpType *xtx, MKL_INT *ldxtx, MKL_INT *ny, algorithmFpType *xty, MKL_INT *ldxty,
                               bool isEqualRidge, algorithmFpType *ridge, algorithmFpType *beta,
                               services::KernelErrorCollection *errors)
{
    MKL_INT n;
    MKL_INT i_one = 1;
    char uplo = 'U';
    MKL_INT info;

    n = (*ny) * (*ldxty);
    daal::services::daal_memcpy_s(beta,n*sizeof(algorithmFpType),xty,n*sizeof(algorithmFpType));

    if (isEqualRidge)
    {
        // X' * X <- X' * X + ridge * I
        algorithmFpType * ptr = xtx;
        for (size_t i = 0; i < static_cast<size_t>(*p); ++i)
        {
            *ptr += *ridge;
            ptr += *p + 1;
        }

        /* Perform L*L' decomposition of X'*X */
        Lapack<algorithmFpType, cpu>::xpotrf( &uplo, p, xtx, ldxtx, &info );
        if ( info < 0 ) { errors->add(services::ErrorRidgeRegressionInternal); return; }
        if ( info > 0 ) { errors->add(services::ErrorRidgeRegressionNormEqSystemSolutionFailed); return; }

        /* Solve L*L'*b=Y */
        Lapack<algorithmFpType, cpu>::xpotrs( &uplo, p, ny, xtx, ldxtx, beta, ldxty, &info );
        if ( info != 0 ) { errors->add(services::ErrorRidgeRegressionInternal); return; }
    }
    else
    {
        const size_t xtxSize = (*p) * (*p);
        const size_t xtxSizeInBytes = xtxSize * sizeof(algorithmFpType);
        algorithmFpType * const tempXTX = services::internal::service_malloc<algorithmFpType, cpu>(xtxSize);
        if (!tempXTX) { errors->add(services::ErrorMemoryAllocationFailed); return; }
        algorithmFpType * betaPtr = beta;
        for (size_t j = 0; j < static_cast<size_t>(*ny); ++j)
        {
            daal::services::daal_memcpy_s(tempXTX, xtxSizeInBytes, xtx, xtxSizeInBytes);
            algorithmFpType * ptr = tempXTX;
            for (size_t i = 0; i < static_cast<size_t>(*p); ++i)
            {
                *ptr += ridge[j];
                ptr += *p + 1;
            }

            /* Perform L*L' decomposition of X'*X */
            Lapack<algorithmFpType, cpu>::xpotrf(&uplo, p, tempXTX, ldxtx, &info);
            if (info < 0) { errors->add(services::ErrorRidgeRegressionInternal); break; }
            if (info > 0) { errors->add(services::ErrorRidgeRegressionNormEqSystemSolutionFailed); break; }

            /* Solve L*L'*b=Y */
            Lapack<algorithmFpType, cpu>::xpotrs(&uplo, p, &i_one, tempXTX, ldxtx, betaPtr, ldxty, &info);
            if (info != 0) { errors->add(services::ErrorRidgeRegressionInternal); break; }

            betaPtr += *ldxty;
        }
        services::internal::service_free<algorithmFpType, cpu>(tempXTX);
    }
}


template <typename algorithmFpType, CpuType cpu>
void updatePartialModelNormEq(NumericTable *x, NumericTable *y,
            ridge_regression::Model *r,
            const daal::algorithms::Parameter *par, bool isOnline,
            services::KernelErrorCollection *errors)
{
    const ridge_regression::Parameter * const parameter = static_cast<const ridge_regression::Parameter *>(par);
    ModelNormEq *rr = static_cast<ModelNormEq *>(r);

    MKL_INT nRows      = (MKL_INT)x->getNumberOfRows();     /* vectors */
    MKL_INT nFeatures  = (MKL_INT)x->getNumberOfColumns();  /* features */
    MKL_INT nResponses = (MKL_INT)y->getNumberOfColumns();  /* variables */
    MKL_INT nBetas     = (MKL_INT)rr->getNumberOfBetas();   /* features + 1 */

    MKL_INT nBetasIntercept = nBetas;
    if (parameter && !parameter->interceptFlag) { nBetasIntercept--; }; /* features + 1 */

    /* Retrieve matrices X'*X and X'*Y from daal::algorithms::Model */
    NumericTable *xtxTable, *xtyTable;
    BlockDescriptor<algorithmFpType> xtxBD, xtyBD;
    algorithmFpType *xtx, *xty;

    getModelPartialSums<algorithmFpType, cpu>(rr, nBetasIntercept, nResponses, readWrite, &xtxTable, xtxBD, &xtx, &xtyTable, xtyBD, &xty);

    /* Retrieve data associated with input tables */
    BlockDescriptor<algorithmFpType> xBD;
    BlockDescriptor<algorithmFpType> yBD;

    x->getBlockOfRows(0, nRows, readOnly, xBD);
    y->getBlockOfRows(0, nRows, readOnly, yBD);

    algorithmFpType *dx = xBD.getBlockPtr();
    algorithmFpType *dy = yBD.getBlockPtr();

    /* Initialize output arrays by zero in case of batch mode */
    if(!isOnline)
    {
        daal::services::internal::service_memset<algorithmFpType, cpu>(xtx, 0, nBetasIntercept * nBetasIntercept);
        daal::services::internal::service_memset<algorithmFpType, cpu>(xty, 0, nResponses * nBetasIntercept);
    }

    /* Split rows by blocks */
    size_t numRowsInBlock = 128;

    size_t numBlocks = nRows / numRowsInBlock;
    if (numBlocks * numRowsInBlock < nRows) { numBlocks++; }

    /* Create TLS xtx buffer */
    daal::tls<algorithmFpType *> xtx_buff( [ = ]()-> algorithmFpType*
    {
      algorithmFpType* ptr = service_scalable_calloc<algorithmFpType, cpu>(nBetasIntercept * nBetasIntercept);
      if (!ptr) { errors->add(services::ErrorMemoryAllocationFailed); }
      return ptr;
    } );

    /* Create TLS xty buffer */
    daal::tls<algorithmFpType *> xty_buff( [ = ]()-> algorithmFpType*
    {
      algorithmFpType* ptr = service_scalable_calloc<algorithmFpType, cpu>(nResponses * nBetasIntercept);
      if (!ptr) { errors->add(services::ErrorMemoryAllocationFailed); }
      return ptr;
    } );

    /* TBB threaded loop */
    daal::threader_for( numBlocks, numBlocks, [ =, &xtx_buff, &xty_buff ](int iBlock)
    {
        algorithmFpType* xtx_local =  xtx_buff.local();
        algorithmFpType* xty_local =  xty_buff.local();

        size_t startRow = iBlock * numRowsInBlock;
        size_t endRow = startRow + numRowsInBlock;
        if (endRow > nRows) { endRow = nRows; }

        algorithmFpType* dx_ptr = dx + startRow * nFeatures;
        algorithmFpType* dy_ptr = dy + startRow * nResponses;

        MKL_INT nP = nFeatures;
        MKL_INT nN = endRow - startRow;
        MKL_INT nB = nBetasIntercept;
        MKL_INT nV = nResponses;

        updatePartialSums<algorithmFpType, cpu>(&nP, &nN, &nB, dx_ptr, xtx_local, &nV, dy_ptr, xty_local);
    } );

    /* Sum all xtx and free buffer */
    xtx_buff.reduce( [ = ](algorithmFpType * v)-> void
    {
       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
          for( size_t i = 0; i < (nBetasIntercept * nBetasIntercept); i++){ xtx[i] += v[i]; }
          service_scalable_free<algorithmFpType, cpu>( v );
    } );

    /* Sum all xty and free buffer */
    xty_buff.reduce( [ = ](algorithmFpType * v)-> void
    {
       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
         for( size_t i = 0; i < (nResponses * nBetasIntercept); i++){ xty[i] += v[i]; }
         service_scalable_free<algorithmFpType, cpu>( v );
    } );

    x->releaseBlockOfRows(xBD);
    y->releaseBlockOfRows(yBD);

    releaseModelNormEqPartialSums<algorithmFpType, cpu>(xtxTable, xtxBD, xtyTable, xtyBD);

} /* updatePartialModelNormEq */


template <typename algorithmFpType, CpuType cpu>
void finalizeModelNormEq(ridge_regression::Model *a, ridge_regression::Model *r, const daal::algorithms::Parameter * par,
                         services::KernelErrorCollection *errors)
{
    ModelNormEq *aa = static_cast<ModelNormEq *>(a);
    ModelNormEq *rr = static_cast<ModelNormEq *>(r);

    MKL_INT nBetas = (MKL_INT)rr->getNumberOfBetas();
    MKL_INT nResponses = (MKL_INT)rr->getNumberOfResponses();
    MKL_INT nBetasIntercept = nBetas;
    if (!rr->getInterceptFlag()) { nBetasIntercept--; }

    algorithmFpType *betaBuffer = (algorithmFpType *)daal::services::daal_malloc(nResponses * nBetas * sizeof(algorithmFpType));
    if (!betaBuffer) { errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Retrieve matrices X'*X and X'*Y from daal::algorithms::Model */
    NumericTable *xtxTable, *xtyTable;
    BlockDescriptor<algorithmFpType> xtxBD, xtyBD, ridgeParamsBD;
    algorithmFpType *xtx, *xty;
    getModelPartialSums<algorithmFpType, cpu>(aa, nBetas, nResponses, readOnly, &xtxTable, xtxBD, &xtx, &xtyTable, xtyBD, &xty);

    const TrainParameter * const trainParameter = static_cast<const TrainParameter *>(par);
    const size_t ridgeParamsNumberOfRows = trainParameter->ridgeParameters->getNumberOfRows();
    trainParameter->ridgeParameters->getBlockOfRows(0, ridgeParamsNumberOfRows, data_management::readOnly, ridgeParamsBD);

    computeRidgeCoeffs<algorithmFpType, cpu>(&nBetasIntercept, xtx, &nBetas, &nResponses, xty, &nBetas, (ridgeParamsNumberOfRows == 1),
                                             ridgeParamsBD.getBlockPtr(), betaBuffer, errors);

    trainParameter->ridgeParameters->releaseBlockOfRows(ridgeParamsBD);

    releaseModelNormEqPartialSums<algorithmFpType, cpu>(xtxTable, xtxBD, xtyTable, xtyBD);

    NumericTable *betaTable = rr->getBeta().get();
    BlockDescriptor<algorithmFpType> betaBD;
    betaTable->getBlockOfRows(0, nResponses, writeOnly, betaBD);
    algorithmFpType *beta = betaBD.getBlockPtr();

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

template <typename algorithmFpType, CpuType cpu>
void RidgeRegressionTrainBatchKernel<algorithmFpType, training::normEqDense, cpu>::compute(
    NumericTable *x, NumericTable *y, ridge_regression::Model *r, const daal::algorithms::Parameter * par)
{
    const bool isOnline = false;
    updatePartialModelNormEq<algorithmFpType, cpu>(x, y, r, par, isOnline, this->_errors.get());
    finalizeModelNormEq<algorithmFpType, cpu>(r, r, par, this->_errors.get());
}

template <typename algorithmfptype, CpuType cpu>
void RidgeRegressionTrainOnlineKernel<algorithmfptype, training::normEqDense, cpu>::compute(
    NumericTable *x, NumericTable *y, ridge_regression::Model *r, const daal::algorithms::Parameter * par)
{
    const bool isOnline = true;
    updatePartialModelNormEq<algorithmfptype, cpu>(x, y, r, par, isOnline, this->_errors.get());
}

template <typename algorithmfptype, CpuType cpu>
void RidgeRegressionTrainOnlineKernel<algorithmfptype, training::normEqDense, cpu>::finalizeCompute(
    ridge_regression::Model *a, ridge_regression::Model *r, const daal::algorithms::Parameter * par)
{
    finalizeModelNormEq<algorithmfptype, cpu>(a, r, par, this->_errors.get());
}

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
