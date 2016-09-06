/* file: linear_regression_train_dense_normeq_impl.i */
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
template<typename interm, CpuType cpu>
static void getModelPartialSums(ModelNormEq *model,
                         MKL_INT dim, MKL_INT ny, ReadWriteMode rwmode,
                         NumericTable **xtxTable, BlockDescriptor<interm> &xtxBD, interm **xtx,
                         NumericTable **xtyTable, BlockDescriptor<interm> &xtyBD, interm **xty)
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
template<typename interm, CpuType cpu>
static void releaseModelNormEqPartialSums(NumericTable *xtxTable, BlockDescriptor<interm> &xtxBD,
                                    NumericTable *xtyTable, BlockDescriptor<interm> &xtyBD)
{
    xtxTable->releaseBlockOfRows(xtxBD);
    xtyTable->releaseBlockOfRows(xtyBD);
}



template <typename interm, CpuType cpu>
static void updatePartialSums(
                       MKL_INT *p,          /* features */
                       MKL_INT *n,          /* vectors */
                       MKL_INT *b,          /* features + 1 */
                       interm  *x_in,       /* p*n input matrix */
                       interm  *xtx_out,    /* p*b output matrix */
                       MKL_INT *v,          /* variables */
                       interm  *y_in,       /* v*n input matrix   */
                       interm  *xty_out     /* v*b output matrix */
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

interm alpha = 1.0;
interm* xtx_ptr;
interm* x_ptr;
interm* y_ptr;

    Blas<interm, cpu>::xxsyrk(&uplo, &trans, p, n, &alpha, x_in, p, &alpha, xtx_out, b);

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

        xtx_ptr[p_val] += (interm)n_val;

    } /* if ( p_val < b_val ) */

    Blas<interm, cpu>::xxgemm(&transa, &transb, p, v, n, &alpha, x_in, p, y_in, v, &alpha, xty_out, b);

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
template <typename interm, CpuType cpu>
static void computeLinregCoeffs(MKL_INT *p,  interm *xtx, MKL_INT *ldxtx,
                     MKL_INT *ny, interm *xty, MKL_INT *ldxty, interm *beta,
                     services::KernelErrorCollection *_errors)
{
    MKL_INT n;
    MKL_INT i_one = 1;
    char uplo = 'U';
    MKL_INT info;

    n = (*ny) * (*ldxty);
    daal::services::daal_memcpy_s(beta,n*sizeof(interm),xty,n*sizeof(interm));

    /* Perform L*L' decomposition of X'*X */
    Lapack<interm, cpu>::xpotrf( &uplo, p, xtx, ldxtx, &info );
    if ( info < 0 ) { _errors->add(services::ErrorLinearRegressionInternal); return; }
    if ( info > 0 ) { _errors->add(services::ErrorNormEqSystemSolutionFailed); return; }

    /* Solve L*L'*b=Y */
    Lapack<interm, cpu>::xpotrs( &uplo, p, ny, xtx, ldxtx, beta, ldxty, &info );
    if ( info != 0 ) { _errors->add(services::ErrorLinearRegressionInternal); return; }

} /* computeLinregCoeffs */


template <typename interm, CpuType cpu>
void updatePartialModelNormEq(NumericTable *x, NumericTable *y,
            linear_regression::Model *r,
            const daal::algorithms::Parameter *par, bool isOnline,
            services::KernelErrorCollection *_errors)
{
    const linear_regression::Parameter *parameter = static_cast<const linear_regression::Parameter *>(par);
    ModelNormEq *rr = static_cast<ModelNormEq *>(r);

    MKL_INT nRows      = (MKL_INT)x->getNumberOfRows();     /* vectors */
    MKL_INT nFeatures  = (MKL_INT)x->getNumberOfColumns();  /* features */
    MKL_INT nResponses = (MKL_INT)y->getNumberOfColumns();  /* variables */
    MKL_INT nBetas     = (MKL_INT)rr->getNumberOfBetas();   /* features + 1 */

    MKL_INT nBetasIntercept = nBetas;
    if (parameter && !parameter->interceptFlag) { nBetasIntercept--; }; /* features + 1 */

    /* Retrieve matrices X'*X and X'*Y from daal::algorithms::Model */
    NumericTable *xtxTable, *xtyTable;
    BlockDescriptor<interm> xtxBD, xtyBD;
    interm *xtx, *xty;

    getModelPartialSums<interm, cpu>(rr, nBetasIntercept, nResponses, readWrite, &xtxTable, xtxBD, &xtx, &xtyTable, xtyBD, &xty);

    /* Retrieve data associated with input tables */
    BlockDescriptor<interm> xBD;
    BlockDescriptor<interm> yBD;

    x->getBlockOfRows(0, nRows, readOnly, xBD);
    y->getBlockOfRows(0, nRows, readOnly, yBD);

    interm *dx = xBD.getBlockPtr();
    interm *dy = yBD.getBlockPtr();

    /* Initialize output arrays by zero in case of batch mode */
    if(!isOnline)
    {
        daal::services::internal::service_memset<interm, cpu>(xtx, 0, nBetasIntercept * nBetasIntercept);
        daal::services::internal::service_memset<interm, cpu>(xty, 0, nResponses * nBetasIntercept);
    }

    /* Split rows by blocks */
    size_t numRowsInBlock = 128;

    size_t numBlocks = nRows / numRowsInBlock;
    if (numBlocks * numRowsInBlock < nRows) { numBlocks++; }

    /* Create TLS xtx buffer */
    daal::tls<interm *> xtx_buff( [ = ]()-> interm*
    {
      interm* ptr = service_scalable_calloc<interm, cpu>(nBetasIntercept * nBetasIntercept);
      if (!ptr) { _errors->add(services::ErrorMemoryAllocationFailed); }
      return ptr;
    } );

    /* Create TLS xty buffer */
    daal::tls<interm *> xty_buff( [ = ]()-> interm*
    {
      interm* ptr = service_scalable_calloc<interm, cpu>(nResponses * nBetasIntercept);
      if (!ptr) { _errors->add(services::ErrorMemoryAllocationFailed); }
      return ptr;
    } );

    /* TBB threaded loop */
    daal::threader_for( numBlocks, numBlocks, [ =, &xtx_buff, &xty_buff ](int iBlock)
    {
        interm* xtx_local =  xtx_buff.local();
        interm* xty_local =  xty_buff.local();

        size_t startRow = iBlock * numRowsInBlock;
        size_t endRow = startRow + numRowsInBlock;
        if (endRow > nRows) { endRow = nRows; }

        interm* dx_ptr = dx + startRow * nFeatures;
        interm* dy_ptr = dy + startRow * nResponses;

        MKL_INT nP = nFeatures;
        MKL_INT nN = endRow - startRow;
        MKL_INT nB = nBetasIntercept;
        MKL_INT nV = nResponses;

        updatePartialSums<interm, cpu>(&nP, &nN, &nB, dx_ptr, xtx_local, &nV, dy_ptr, xty_local);
    } );

    /* Sum all xtx and free buffer */
    xtx_buff.reduce( [ = ](interm * v)-> void
    {
       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
          for( size_t i = 0; i < (nBetasIntercept * nBetasIntercept); i++){ xtx[i] += v[i]; }
          service_scalable_free<interm, cpu>( v );
    } );

    /* Sum all xty and free buffer */
    xty_buff.reduce( [ = ](interm * v)-> void
    {
       PRAGMA_IVDEP
       PRAGMA_VECTOR_ALWAYS
         for( size_t i = 0; i < (nResponses * nBetasIntercept); i++){ xty[i] += v[i]; }
         service_scalable_free<interm, cpu>( v );
    } );

    x->releaseBlockOfRows(xBD);
    y->releaseBlockOfRows(yBD);

    releaseModelNormEqPartialSums<interm, cpu>(xtxTable, xtxBD, xtyTable, xtyBD);

} /* updatePartialModelNormEq */


template <typename interm, CpuType cpu>
void finalizeModelNormEq(linear_regression::Model *a, linear_regression::Model *r,
                   services::KernelErrorCollection *_errors)
{
    ModelNormEq *aa = static_cast<ModelNormEq *>(a);
    ModelNormEq *rr = static_cast<ModelNormEq *>(r);

    MKL_INT nBetas = (MKL_INT)rr->getNumberOfBetas();
    MKL_INT nResponses = (MKL_INT)rr->getNumberOfResponses();
    MKL_INT nBetasIntercept = nBetas;
    if (!rr->getInterceptFlag()) { nBetasIntercept--; }

    interm *betaBuffer = (interm *)daal::services::daal_malloc(nResponses * nBetas * sizeof(interm));
    if (!betaBuffer) { _errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Retrieve matrices X'*X and X'*Y from daal::algorithms::Model */
    NumericTable *xtxTable, *xtyTable;
    BlockDescriptor<interm> xtxBD, xtyBD;
    interm *xtx, *xty;
    getModelPartialSums<interm, cpu>(aa, nBetas, nResponses, readOnly, &xtxTable, xtxBD, &xtx, &xtyTable, xtyBD, &xty);

    computeLinregCoeffs<interm, cpu>(&nBetasIntercept, xtx, &nBetas, &nResponses, xty, &nBetas, betaBuffer, _errors);

    releaseModelNormEqPartialSums<interm, cpu>(xtxTable, xtxBD, xtyTable, xtyBD);

    NumericTable *betaTable = rr->getBeta().get();
    BlockDescriptor<interm> betaBD;
    betaTable->getBlockOfRows(0, nResponses, writeOnly, betaBD);
    interm *beta = betaBD.getBlockPtr();

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


template <typename interm, CpuType cpu>
void LinearRegressionTrainBatchKernel<interm, training::normEqDense, cpu>::compute(
            NumericTable *x, NumericTable *y, linear_regression::Model *r,
             const daal::algorithms::Parameter *par)
{
    bool isOnline = false;
    updatePartialModelNormEq<interm, cpu>(x, y, r, par, isOnline, this->_errors.get());
    finalizeModelNormEq<interm, cpu>(r, r, this->_errors.get());
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainOnlineKernel<interm, training::normEqDense, cpu>::compute(
            NumericTable *x, NumericTable *y, linear_regression::Model *r,
             const daal::algorithms::Parameter *par)
{
    bool isOnline = true;
    updatePartialModelNormEq<interm, cpu>(x, y, r, par, isOnline, this->_errors.get());
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainOnlineKernel<interm, training::normEqDense, cpu>::finalizeCompute(
            linear_regression::Model *a, linear_regression::Model *r,
            const daal::algorithms::Parameter *par)
{
    finalizeModelNormEq<interm, cpu>(a, r, this->_errors.get());
}

} /* namespace internal */
} /* namespace training */
} /* namespace linear_regression */
} /* namespace algorithms */
} /* namespace daal */

#endif
