/* file: linear_regression_train_dense_qr_impl.i */
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
//  Implementation of auxiliary functions for linear regression qrDense method.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_QR_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_QR_IMPL_I__

#include "threading.h"
#include "service_lapack.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "linear_regression_train_kernel.h"
#include "algorithms/linear_regression/linear_regression_qr_model.h"

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
 *  \param  rTable[out]   Numeric table containing matrix R
 *  \param  rBD[out]      Buffer manager corresponding to rTable
 *  \param  r[out]        Array containing matrix R
 *  \param  qtyTable[out] Numeric table containing matrix Q'*Y
 *  \param  qtyBD[out]    Buffer manager corresponding to qtyTable
 *  \param  qty[out]      Array containing matrix Q'*Y
 */
template <typename algorithmFPType, CpuType cpu>
static void getModelPartialSums(ModelQR *model,
                                DAAL_INT dim, DAAL_INT ny, ReadWriteMode rwmode,
                                NumericTable **rTable,   BlockDescriptor<algorithmFPType> &rBD,   algorithmFPType **r,
                                NumericTable **qtyTable, BlockDescriptor<algorithmFPType> &qtyBD, algorithmFPType **qty)
{
    *rTable   = model->getRTable().get();
    *qtyTable = model->getQTYTable().get();

    (*  rTable)->getBlockOfRows(0, dim, rwmode, rBD);
    (*qtyTable)->getBlockOfRows(0, ny,  rwmode, qtyBD);

    *r   = rBD  .getBlockPtr();
    *qty = qtyBD.getBlockPtr();
}

/**
 *  \brief Release arrays holding partial sums in Linear Regression daal::algorithms::Model
 *
 *  \param  rTable[in]    Numeric table containing matrix R
 *  \param  rBD[in]       Buffer manager corresponding to rTable
 *  \param  qtyTable[in]  Numeric table containing matrix Q'*Y
 *  \param  qtyBD[in]     Buffer manager corresponding to qtyTable
 */
template <typename algorithmFPType, CpuType cpu>
static void releaseModelQRPartialSums(NumericTable *rTable,   BlockDescriptor<algorithmFPType> &rBD,
                                      NumericTable *qtyTable, BlockDescriptor<algorithmFPType> &qtyBD)
{
    rTable->releaseBlockOfRows(rBD);
    qtyTable->releaseBlockOfRows(qtyBD);
}

/**
 *  \brief Calculate size of LAPACK WORK buffer needed to perform qrDense decomposition
 *
 *  \param p[in]        Number of columns in input matrix
 *  \param n[in]        Number of rows in input matrix
 *  \param x[in]        Input matrix of size (n x p), n > p
 *  \param tau[in]      LAPACK GERQF TAU parameter. Array of size p
 *  \param work[in]     LAPACK GERQF WORK parameter
 *  \param lwork[out]   Calculated size of WORK array
 *
 */

template <typename algorithmFPType, CpuType cpu>
static void computeQRWorkSize(DAAL_INT *p, DAAL_INT *n, algorithmFPType *x, algorithmFPType *tau, DAAL_INT *lwork,
                              services::KernelErrorCollection *_errors)
{
    DAAL_INT info = 0;
    algorithmFPType work_local;

    *lwork = -1;
    Lapack<algorithmFPType, cpu>::xxgerqf(p, n, x, p, tau, &work_local, lwork, &info);


    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }

    *lwork = (DAAL_INT)work_local;
}

/**
 *  \brief Function that allocates memory for storing intermediate data
 *         for qrDense decomposition
 *
 *  \param p[in]        Number of columns in input matrix X
 *  \param n[in]        Number of rows in input matrix X
 *  \param x[in]        Input matrix X of size (n x p), n > p
 *  \param ny[in]       Number of columns in input matrix Y
 *  \param y[in]        Input matrix Y of size (n x ny)
 *  \param tau[in]      LAPACK GERQF/ORMRQ TAU parameter. Array of size p
 *  \param work[out]    LAPACK GERQF/ORMRQ WORK parameter
 *  \param lwork[out]   Calculated size of WORK array
 *
 */
template <typename algorithmFPType, CpuType cpu>
static void mallocQRWorkBuffer(DAAL_INT *p, DAAL_INT *n, algorithmFPType *x, DAAL_INT *ny, algorithmFPType *y, algorithmFPType *tau,
                               TArrayScalable<algorithmFPType, cpu> &work, DAAL_INT *lwork, services::KernelErrorCollection *_errors)
{
    DAAL_INT info = 0;
    DAAL_INT lwork1;

    computeQRWorkSize<algorithmFPType, cpu>(p, n, x, tau, &lwork1, _errors);
    if (!_errors->isEmpty()) { return; }

    char side = 'R';
    char trans = 'T';
    DAAL_INT lwork2 = -1;
    algorithmFPType work_local;

    Lapack<algorithmFPType, cpu>::xxormrq(&side, &trans, ny, n, p, x, p, tau, y, ny, &work_local, &lwork2, &info);

    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }

    lwork2 = (DAAL_INT)work_local;
    *lwork = ((lwork1 > lwork2) ? lwork1 : lwork2);

    work.reset(*lwork);
    if (!work.get()) { _errors->add(services::ErrorMemoryAllocationFailed); return; }

}

/**
 *  \brief Function that copies input matrices X and Y into intermediate
 *         buffers.
 *
 *  \param dim[in]       Number of columns in input matrix X
 *  \param betaDim[in]   Number of regression coefficients
 *  \param n[in]         Number of rows in input matrix X
 *  \param x[in]         Input matrix X of size (n x p), n > p
 *  \param ny[in]        Number of columns in input matrix Y
 *  \param y[in]         Input matrix Y of size (n x ny)
 *  \param qrBuffer[out] if (dim     == betaDim) copy of matrix X,
 *                       if (dim + 1 == betaDim) qrBuffer = (X|e),
 *                          where e is a column vector of all 1's.
 *  \param qtyBuffer[out] copy of matrix Y
 *
 */
template <typename algorithmFPType, CpuType cpu>
static void copyDataToBuffer(DAAL_INT *dim, DAAL_INT *betaDim, DAAL_INT *n, algorithmFPType *x, DAAL_INT *ny, algorithmFPType *y,
                             algorithmFPType *qrBuffer, algorithmFPType *qtyBuffer)
{
    DAAL_INT iOne = 1;             // integer one
    DAAL_INT dimVal = *dim;
    DAAL_INT betaDimVal = *betaDim;
    DAAL_INT nVal = *n;
    DAAL_INT ySize = (*ny) * nVal;

    /* Copy matrix X to temporary buffer in order not to damage it */
    if (dimVal == betaDimVal)
    {
        DAAL_INT xSize = dimVal * nVal;
        daal::services::daal_memcpy_s(qrBuffer, xSize * sizeof(algorithmFPType), x, xSize * sizeof(algorithmFPType));
    }
    else
    {
        for (size_t i = 0; i < nVal; i++)
        {
            daal::services::daal_memcpy_s(qrBuffer + i * betaDimVal, dimVal * sizeof(algorithmFPType), x + i * dimVal, dimVal * sizeof(algorithmFPType));
            qrBuffer[i * betaDimVal + betaDimVal - 1] = 1.0;
        }
    }

    /* Copy matrix Y to temporary buffer in order not to damage it */
    daal::services::daal_memcpy_s(qtyBuffer, ySize * sizeof(algorithmFPType), y, ySize * sizeof(algorithmFPType));
}

/**
 *  \brief Function that calculates R and Y*Q' from input matrix
 *         of independent variables X and matrix of responses Y.
 *
 *  \param p[in]     Number of columns in input matrix X
 *  \param n[in]     Number of rows in input matrix X
 *  \param x[in,out] Input matrix X of size (n x p), n > p;
 *                   Overwritten by LAPACK on output
 *  \param ny[in]    Number of columns in input matrix Y
 *  \param y[in,out] Input matrix Y of size (n x ny);
 *                   Overwritten by LAPACK on output
 *  \param r[out]    Matrix R of size (p x p)
 *  \param qty[out]  Matrix Y*Q' of size (ny x p)
 *  \param tau[in]   LAPACK GERQF/ORMRQ TAU parameter. Array of size p
 *  \param work[in]  LAPACK GERQF/ORMRQ WORK parameter
 *  \param lwork[in] Calculated size of WORK array
 *
 */
template <typename algorithmFPType, CpuType cpu>
static void computeQRForBlock(DAAL_INT *p, DAAL_INT *n, algorithmFPType *x, DAAL_INT *ny, algorithmFPType *y, algorithmFPType *r,
                              algorithmFPType *qty,
                              algorithmFPType *tau, algorithmFPType *work, DAAL_INT *lwork, services::KernelErrorCollection *_errors)
{
    DAAL_INT iOne = 1;             // integer one
    DAAL_INT info = 0;
    DAAL_INT pVal = *p;
    DAAL_INT n_val = *n;
    DAAL_INT ny_val = *ny;
    DAAL_INT qtySize = ny_val * pVal;
    DAAL_INT rOffset = (n_val - pVal) * pVal;
    DAAL_INT yqtOffset = (n_val - pVal) * ny_val;

    /* Calculate RQ decomposition of X */
    Lapack<algorithmFPType, cpu>::xxgerqf(p, n, x, p, tau, work, lwork, &info);

    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }

    /* Copy result into matrix R */
    algorithmFPType *xPtr = x + rOffset;
    for (size_t i = 0; i < pVal; i++)
    {
      PRAGMA_IVDEP
      PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j <= i; j++)
        {
            r[i * pVal + j] = xPtr[i * pVal + j];
        }
    }

    /* Calculate Y*Q' */
    char side = 'R';
    char trans = 'T';
    Lapack<algorithmFPType, cpu>::xxormrq(&side, &trans, ny, n, p, x, p, tau, y, ny, work, lwork, &info);
    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }

    /* Copy result into matrix QTY */
    daal::services::daal_memcpy_s(qty, qtySize * sizeof(algorithmFPType), y + yqtOffset, qtySize * sizeof(algorithmFPType));
}



/**
 *  \brief Function that merges qrDense partial sums (R1, QTY1), (R2, QTY2)
 *         into partial sum (R, QTY).
 *
 *  \param p[in]     Number of rows and columns in R, R1, R2 and
 *                   number of rows in QTY, QTY1, QTY2.
 *  \param ny[in]    Number of columns in QTY, QTY1, QTY2.
 *  \param r1[in]    Matrix of size (p x p)
 *  \param qty1[in]  Matrix of size (p x ny)
 *  \param r2[in]    Matrix of size (p x p)
 *  \param qty2[in]  Matrix of size (p x ny)
 *  \param r12[in]   Matrix of size (2p x p)
 *  \param qty12[in] Matrix of size (2p x ny)
 *  \param r[out]    Output matrix of size (p x p)
 *  \param qty[out]  Output matrix of size (p x ny)
 *  \param tau[in]   LAPACK GERQF TAU parameter. Array of size p
 *  \param work[in]  LAPACK GERQF WORK parameter
 *  \param lwork[in] Size of WORK array
 *
 */
template <typename algorithmFPType, CpuType cpu>
static void mergeQR(DAAL_INT *p, DAAL_INT *ny, algorithmFPType *r1,  algorithmFPType *qty1,  algorithmFPType *r2, algorithmFPType *qty2,
                    algorithmFPType *r12, algorithmFPType *qty12, algorithmFPType *r,  algorithmFPType *qty, algorithmFPType *tau, algorithmFPType *work, DAAL_INT *lwork,
                    services::KernelErrorCollection *_errors)
{
    DAAL_INT iOne = 1;             // integer one
    DAAL_INT p_val = *p;
    DAAL_INT n_val = 2 * p_val;
    DAAL_INT ny_val = *ny;
    DAAL_INT rSize = p_val * p_val;
    DAAL_INT qtySize = p_val * ny_val;

    /* Copy R1 and R2 into R12. R12 = (R1, R2) */
    daal::services::daal_memcpy_s(r12        , rSize * 2 * sizeof(algorithmFPType), r1, rSize * sizeof(algorithmFPType));
    daal::services::daal_memcpy_s(r12 + rSize, rSize * sizeof(algorithmFPType),  r2, rSize * sizeof(algorithmFPType));
    /* Copy QTY1 and QTY2 into QTY12. QTY12 = (QTY1, QTY2) */
    daal::services::daal_memcpy_s(qty12          , qtySize * 2 * sizeof(algorithmFPType), qty1, qtySize * sizeof(algorithmFPType));
    daal::services::daal_memcpy_s(qty12 + qtySize, qtySize * sizeof(algorithmFPType),  qty2, qtySize * sizeof(algorithmFPType));

    computeQRForBlock<algorithmFPType, cpu>(p, &n_val, r12, ny, qty12, r, qty, tau, work, lwork, _errors);
}


/**
 *  \brief Function that computes linear regression coefficients
 *         from partial sums (R, QTY).
 *
 *  \param p[in]     Number of regression coefficients
 *  \param r[in]     Matrix of size (p x p)
 *  \param ny[in]    Number of dependent variables
 *  \param qty[in]   Matrix of size (p x ny)
 *  \param beta[out] Matrix of regression coefficients of size (ny x p)
 *
 */
template <typename algorithmFPType, CpuType cpu>
static void finalizeQR(DAAL_INT *p, algorithmFPType *r, DAAL_INT *ny, algorithmFPType *qty, algorithmFPType *beta,
                       services::KernelErrorCollection *_errors)
{
    DAAL_INT iOne = 1;             // integer one
    DAAL_INT info = 0;
    DAAL_INT betaSize = (*ny) * (*p);
    DAAL_INT pVal = *p;
    DAAL_INT ny_val = *ny;

    for (size_t i = 0; i < ny_val; i++)
    {
      PRAGMA_IVDEP
      PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < pVal; j++)
        {
            beta[i * pVal + j] = qty[j * ny_val + i];
        }
    }

    /* Solve triangular linear system R'*beta = Y*Q' */
    char uplo = 'U';
    char trans = 'T';
    char diag = 'N';
    Lapack<algorithmFPType, cpu>::xtrtrs(&uplo, &trans, &diag, p, ny, r, p, beta, p, &info);
    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }
}

template <typename algorithmFPType, CpuType cpu>
void updatePartialModelQR(NumericTable *x, NumericTable *y,
                          linear_regression::Model *r,
                          const daal::algorithms::Parameter *par, bool isOnline,
                          services::KernelErrorCollection *_errors)
{
    const linear_regression::Parameter *parameter = static_cast<const linear_regression::Parameter *>(par);
    ModelQR *rr = static_cast<ModelQR *>(r);
    DAAL_INT nFeatures = (DAAL_INT)x->getNumberOfColumns();
    DAAL_INT nResponses = (DAAL_INT)y->getNumberOfColumns();
    DAAL_INT nRows = (DAAL_INT)x->getNumberOfRows();
    DAAL_INT nBetas = (DAAL_INT)rr->getNumberOfBetas();
    DAAL_INT nBetasIntercept = nBetas;
    if (parameter && !parameter->interceptFlag) { nBetasIntercept--; }

    /* Retrieve data associated with input tables */
    BlockDescriptor<algorithmFPType> xBD;
    BlockDescriptor<algorithmFPType> yBD;
    x->getBlockOfRows(0, nRows, readOnly, xBD);
    y->getBlockOfRows(0, nRows, readOnly, yBD);
    algorithmFPType *dy = yBD.getBlockPtr();
    algorithmFPType *dx = xBD.getBlockPtr();

    /* Retrieve matrices R and Q'*Y from daal::algorithms::Model */
    NumericTable *rTable, *qtyTable;
    BlockDescriptor<algorithmFPType> rBD, qtyBD;
    algorithmFPType *qrR, *qrQTY;
    getModelPartialSums<algorithmFPType, cpu>(rr, nBetasIntercept, nResponses, readWrite, &rTable, rBD, &qrR, &qtyTable, qtyBD, &qrQTY);

    TArrayScalable<algorithmFPType, cpu> qrROld, qrQTYOld, qrRMerge, qrQTYMerge;
    if (isOnline)
    {
        qrROld    .reset(nBetasIntercept * nBetasIntercept );
        qrQTYOld  .reset(nBetasIntercept * nResponses );
        qrRMerge  .reset(2 * nBetasIntercept * nBetasIntercept );
        qrQTYMerge.reset(2 * nBetasIntercept * nResponses );
        if (!qrROld.get() || !qrQTYOld.get() || !qrRMerge.get() || !qrQTYMerge.get())
        { _errors->add(services::ErrorMemoryAllocationFailed); return; }

        daal::services::daal_memcpy_s(qrROld.get(), nBetasIntercept * nBetasIntercept * sizeof(algorithmFPType), qrR,
                                      nBetasIntercept * nBetasIntercept * sizeof(algorithmFPType));
        daal::services::daal_memcpy_s(qrQTYOld.get(), nBetasIntercept * nResponses * sizeof(algorithmFPType), qrQTY,
                                      nBetasIntercept * nResponses * sizeof(algorithmFPType));
    }

    DAAL_INT lwork = -1;
    TArrayScalable<algorithmFPType, cpu> tau(nBetasIntercept );
    TArrayScalable<algorithmFPType, cpu> qrBuffer(nBetasIntercept * nRows );
    TArrayScalable<algorithmFPType, cpu> qtyBuffer(nResponses * nRows );
    TArrayScalable<algorithmFPType, cpu> work;
    if (!tau.get() || !qrBuffer.get() || !qtyBuffer.get())
    { _errors->add(services::ErrorMemoryAllocationFailed); return; }

    mallocQRWorkBuffer<algorithmFPType, cpu>(&nBetasIntercept, &nRows, dx, &nResponses, dy, tau, work, &lwork, _errors);
    if(!_errors->isEmpty()) { return; }

    copyDataToBuffer<algorithmFPType, cpu>(&nFeatures, &nBetasIntercept, &nRows, dx, &nResponses, dy, qrBuffer, qtyBuffer);

    computeQRForBlock<algorithmFPType, cpu>(&nBetasIntercept, &nRows, qrBuffer, &nResponses, qtyBuffer, qrR, qrQTY,
                                            tau, work, &lwork, _errors);
    if(!_errors->isEmpty()) { return; }

    x->releaseBlockOfRows(xBD);
    y->releaseBlockOfRows(yBD);

    if (isOnline)
    {
        mergeQR<algorithmFPType, cpu>(&nBetasIntercept, &nResponses, qrR, qrQTY, qrROld.get(), qrQTYOld.get(),
                                      qrRMerge.get(), qrQTYMerge.get(), qrR, qrQTY, tau, work, &lwork, _errors);
        if(!_errors->isEmpty()) { return; }
    }

    releaseModelQRPartialSums<algorithmFPType, cpu>(rTable, rBD, qtyTable, qtyBD);
}


template<typename algorithmFPType, CpuType cpu>
struct tls_task_t
{
    DAAL_NEW_DELETE();

    tls_task_t(size_t nBetasIntercept, size_t nRowsInLastBlock, size_t nResponses) :
        work(0), lwork(-1), memAllocError(false),
        tau         (nBetasIntercept),
        qrBuffer    (nBetasIntercept * nRowsInLastBlock),
        qtyBuffer   (nResponses * nRowsInLastBlock),
        qrR         (nBetasIntercept * nBetasIntercept, true),
        qrQTY       (nBetasIntercept * nResponses,      true),
        qrRNew      (nBetasIntercept * nBetasIntercept, true),
        qrQTYNew    (nBetasIntercept * nResponses,      true),
        qrRMerge    (2 * nBetasIntercept * nBetasIntercept),
        qrQTYMerge  (2 * nBetasIntercept * nResponses)
    {
        if (!tau.get() || !qrBuffer.get() || !qtyBuffer.get() || !qrR.get() || !qrQTY.get() ||
            !qrRNew.get() || !qrQTYNew.get() || !qrRMerge.get() || !qrQTYMerge.get())
        {
            memAllocError = true;
        }
    }

    ~tls_task_t()
    {}

    TArrayScalable<algorithmFPType, cpu> work;
    TArrayScalable<algorithmFPType, cpu> tau;
    TArrayScalable<algorithmFPType, cpu> qrBuffer;
    TArrayScalable<algorithmFPType, cpu> qtyBuffer;

    TArrayScalable<algorithmFPType, cpu> qrR;
    TArrayScalable<algorithmFPType, cpu> qrQTY;

    TArrayScalable<algorithmFPType, cpu> qrRNew;
    TArrayScalable<algorithmFPType, cpu> qrQTYNew;
    TArrayScalable<algorithmFPType, cpu> qrRMerge;
    TArrayScalable<algorithmFPType, cpu> qrQTYMerge;

    DAAL_INT lwork;
    bool memAllocError;
};


template <typename algorithmFPType, CpuType cpu>
void updatePartialModelQR_threaded(
            NumericTable *x,
            NumericTable *y,
            linear_regression::Model *r,
            const daal::algorithms::Parameter *par,
            bool isOnline,
            services::KernelErrorCollection *_errors
           )
{
    const linear_regression::Parameter *parameter = static_cast<const linear_regression::Parameter *>(par);
    ModelQR *rr = static_cast<ModelQR *>(r);

    DAAL_INT nFeatures = (DAAL_INT)x->getNumberOfColumns();
    DAAL_INT nResponses = (DAAL_INT)y->getNumberOfColumns();
    DAAL_INT nRows = (DAAL_INT)x->getNumberOfRows();
    DAAL_INT nBetas = (DAAL_INT)rr->getNumberOfBetas();
    DAAL_INT nBetasIntercept = nBetas;
    if (parameter && !parameter->interceptFlag) { nBetasIntercept--; }

    /* Retrieve data associated with input tables */
    BlockDescriptor<algorithmFPType> xBD;
    BlockDescriptor<algorithmFPType> yBD;

    x->getBlockOfRows(0, nRows, readOnly, xBD);
    y->getBlockOfRows(0, nRows, readOnly, yBD);

    algorithmFPType *dy = yBD.getBlockPtr();
    algorithmFPType *dx = xBD.getBlockPtr();

    /* Split rows by blocks */
#if ( __CPUID__(DAAL_CPU) == __avx512_mic__ )
    size_t nDefaultBlockSize = (nRows<=10000)?1024:((nRows>=1000000)?512:2048);
#else
    size_t nDefaultBlockSize = 128;
#endif

    // Block size cannot be bigger than nRows
    size_t nRowsInBlock = (nRows>nDefaultBlockSize)?nDefaultBlockSize:nRows;
    // Block size cannot be smaller than nFeatures+1
    nRowsInBlock = (nRowsInBlock<(nFeatures+1))?(nFeatures+1):nRowsInBlock;

    size_t nBlocks = nRows / nRowsInBlock;
    size_t nRowsInLastBlock = nRowsInBlock + (nRows - nBlocks * nRowsInBlock);

    /* Retrieve matrices R and Q'*Y from daal::algorithms::Model */
    NumericTable *rTable, *qtyTable;
    BlockDescriptor<algorithmFPType> rBD, qtyBD;
    algorithmFPType *qrR, *qrQTY;
    getModelPartialSums<algorithmFPType, cpu>(rr, nBetasIntercept, nResponses, readWrite, &rTable, rBD, &qrR, &qtyTable, qtyBD, &qrQTY);
    if(!isOnline)
    {
        service_memset<algorithmFPType, cpu>(qrR,   (algorithmFPType)0, nBetasIntercept * nBetasIntercept);
        service_memset<algorithmFPType, cpu>(qrQTY, (algorithmFPType)0, nBetasIntercept * nResponses);
    }

    daal::tls<tls_task_t<algorithmFPType, cpu>*> tls_task( [ = ]()->tls_task_t<algorithmFPType, cpu> *
    {
        tls_task_t<algorithmFPType, cpu> *tt_local = new tls_task_t<algorithmFPType, cpu>(nBetasIntercept, nRowsInLastBlock, nResponses);
        if (!tt_local) { _errors->add(services::ErrorMemoryAllocationFailed); return 0; }

        if (tt_local->memAllocError)
        {
            delete tt_local;
            _errors->add(services::ErrorMemoryAllocationFailed);
            return 0;
        }

        algorithmFPType *dx_local = dx;
        algorithmFPType *dy_local = dy;

        DAAL_INT nBetasIntercept_local = nBetasIntercept;
        DAAL_INT nResponses_local      = nResponses;
        DAAL_INT nRowsInLastBlock_local       = nRowsInLastBlock;

        services::KernelErrorCollection *_errors_local = _errors;

        // Function that allocates memory for storing intermediate data for qrDense decomposition
        mallocQRWorkBuffer<algorithmFPType, cpu>(&nBetasIntercept_local,  // in      Number of columns in input matrix X
                                                 &nRowsInLastBlock_local, // in      Number of rows in input matrix X
                                                 dx_local,                // in      Input matrix X of size (n x p), n > p
                                                 &nResponses_local,       // in      Number of columns in input matrix Y
                                                 dy_local,                // in      Input matrix Y of size (n x ny)
                                                 tt_local->tau.get(),     // in      LAPACK GERQF/ORMRQ TAU parameter. Array of size p
                                                 tt_local->work,          // out     LAPACK GERQF/ORMRQ WORK parameter
                                                 & (tt_local->lwork),     // out     Calculated size of WORK array
                                                 _errors_local); if(!_errors_local->isEmpty()) { delete tt_local; return 0; }

        return tt_local;
    } ); /* Allocate memory for all arrays inside TLS: end */


    daal::threader_for( nBlocks, nBlocks, [=, &tls_task](int iBlock)
    {
        struct tls_task_t<algorithmFPType, cpu> *tt_local = tls_task.local();
        if (tt_local)
        {
            algorithmFPType *work_local       = tt_local->work.get();
            DAAL_INT *plwork_local    = &(tt_local->lwork);

            algorithmFPType *tau_local        = tt_local->tau.get();
            algorithmFPType *qrBuffer_local   = tt_local->qrBuffer.get();
            algorithmFPType *qtyBuffer_local  = tt_local->qtyBuffer.get();

            algorithmFPType *qrR_local        = tt_local->qrR.get();
            algorithmFPType *qrQTY_local      = tt_local->qrQTY.get();

            algorithmFPType *qrRNew_local     = tt_local->qrRNew.get();
            algorithmFPType *qrQTYNew_local   = tt_local->qrQTYNew.get();

            algorithmFPType *qrRMerge_local   = tt_local->qrRMerge.get();
            algorithmFPType *qrQTYMerge_local = tt_local->qrQTYMerge.get();

            DAAL_INT nBetasIntercept_local = nBetasIntercept;
            DAAL_INT nResponses_local      = nResponses;
            DAAL_INT nFeatures_local       = nFeatures;

            services::KernelErrorCollection *_errors_local = _errors;

            size_t startRow = iBlock * nRowsInBlock;
            size_t nCurrentRowsInBlock = (iBlock < (nBlocks-1))?nRowsInBlock:nRowsInLastBlock;
            size_t endRow = startRow + nCurrentRowsInBlock;
            DAAL_INT nN_local = endRow - startRow;

            algorithmFPType *dx_local = dx + startRow * nFeatures;
            algorithmFPType *dy_local = dy + startRow * nResponses;


            // Function that copies input matrices X and Y into intermediate buffers.
            copyDataToBuffer<algorithmFPType, cpu>(  &nFeatures_local,        // in      Number of columns in input matrix X
                                                     &nBetasIntercept_local,  // in      Number of regression coefficients
                                                     &nN_local,               // in      Number of rows in input matrix X
                                                     dx_local,                // in      Input matrix X of size (n x p), n > p
                                                     &nResponses_local,       // in      Number of columns in input matrix Y
                                                     dy_local,                // in      Input matrix Y of size (n x ny)
                                                     qrBuffer_local,          // out     if (dim == betaDim) copy of matrix X, if (dim + 1 == betaDim) qrBuffer = (X|e) where e is a column vector of all 1's.
                                                     qtyBuffer_local);        // out     copy of matrix Y

            // Function that calculates R and Y*Q' from input matrix  of independent variables X and matrix of responses Y.
            computeQRForBlock<algorithmFPType, cpu>( &nBetasIntercept_local,  // in      Number of columns in input matrix X
                                                     &nN_local,               // in      Number of rows in input matrix X
                                                     qrBuffer_local,          // in, out Input matrix X of size (n x p), n > p; Overwritten by LAPACK on output
                                                     &nResponses_local,       // in      Number of columns in input matrix Y
                                                     qtyBuffer_local,         // in, out Input matrix Y of size (n x ny); Overwritten by LAPACK on output
                                                     qrRNew_local,            // out     Matrix R of size (p x p)
                                                     qrQTYNew_local,          // out     Matrix Y*Q' of size (ny x p)
                                                     tau_local,               // in      LAPACK GERQF/ORMRQ TAU parameter. Array of size p
                                                     work_local,              // in      LAPACK GERQF/ORMRQ WORK parameter
                                                     plwork_local,            // in      Calculated size of WORK array
                                                     _errors_local); if(!_errors_local->isEmpty()) { return; }

            // Function that merges qrDense partial sums (R1, QTY1), (R2, QTY2) into partial sum (R, QTY)
            mergeQR<algorithmFPType, cpu>(           &nBetasIntercept_local,  // in      Number of rows and columns in R, R1, R2
                                                     &nResponses_local,       // in      Number of columns in QTY, QTY1, QTY2.
                                                     qrRNew_local,            // in      Matrix of size (p x p)
                                                     qrQTYNew_local,          // in      Matrix of size (p x ny)
                                                     qrR_local,               // in      Matrix of size (p x p)
                                                     qrQTY_local,             // in      Matrix of size (p x ny)
                                                     qrRMerge_local,          // in      Matrix of size (2p x p)
                                                     qrQTYMerge_local,        // in      Matrix of size (2p x ny)
                                                     qrR_local,               // out     Output matrix of size (p x p)
                                                     qrQTY_local,             // out     Output matrix of size (p x ny)
                                                     tau_local,               // in      LAPACK GERQF TAU parameter. Array of size p
                                                     work_local,              // in      LAPACK GERQF WORK parameter
                                                     plwork_local,            // in      Size of WORK array
                                                     _errors_local); if(!_errors_local->isEmpty()) { return; }
        }
    } ); /* for(iBlock = 0; iBlock < nBlocks; iBlock++) */


    tls_task.reduce( [ = ](tls_task_t<algorithmFPType, cpu> *tt_local)-> void
    {
        if (tt_local)
        {
            algorithmFPType *work_local       = tt_local->work.get();
            DAAL_INT *plwork_local            = &(tt_local->lwork);

            algorithmFPType *tau_local        = tt_local->tau.get();

            algorithmFPType *qrR_local        = tt_local->qrR.get();
            algorithmFPType *qrQTY_local      = tt_local->qrQTY.get();

            algorithmFPType *qrRMerge_local   = tt_local->qrRMerge.get();
            algorithmFPType *qrQTYMerge_local = tt_local->qrQTYMerge.get();

            DAAL_INT nBetasIntercept_local = nBetasIntercept;
            DAAL_INT nResponses_local      = nResponses;
            DAAL_INT nFeatures_local       = nFeatures;
            services::KernelErrorCollection *_errors_local = _errors;

            // Function that merges qrDense partial sums (R1, QTY1), (R2, QTY2) into partial sum (R, QTY)
            mergeQR<algorithmFPType, cpu>(           &nBetasIntercept_local,  // in      Number of rows and columns in R, R1, R2
                                                     &nResponses_local,       // in      Number of columns in QTY, QTY1, QTY2.
                                                     qrR_local,               // in      Matrix of size (p x p)
                                                     qrQTY_local,             // in      Matrix of size (p x ny)
                                                     qrR,                     // in      Matrix of size (p x p)
                                                     qrQTY,                   // in      Matrix of size (p x ny)
                                                     qrRMerge_local,          // in      Matrix of size (2p x p)
                                                     qrQTYMerge_local,        // in      Matrix of size (2p x ny)
                                                     qrR,                     // out     Output matrix of size (p x p)
                                                     qrQTY,                   // out     Output matrix of size (p x ny)
                                                     tau_local,               // in      LAPACK GERQF TAU parameter. Array of size p
                                                     work_local,              // in      LAPACK GERQF WORK parameter
                                                     plwork_local,            // in      Size of WORK array
                                                     _errors_local); if(!_errors_local->isEmpty()) { return; }

            delete tt_local;
        }
    } );

    x->releaseBlockOfRows(xBD);
    y->releaseBlockOfRows(yBD);

    releaseModelQRPartialSums<algorithmFPType, cpu>(rTable, rBD, qtyTable, qtyBD);
}




template <typename algorithmFPType, CpuType cpu>
void finalizeModelQR(linear_regression::Model *a, linear_regression::Model *r,
                   services::KernelErrorCollection *_errors)
{
    ModelQR *aa = static_cast<ModelQR *>(a);
    ModelQR *rr = static_cast<ModelQR *>(r);

    DAAL_INT nBetas = (DAAL_INT)rr->getNumberOfBetas();
    DAAL_INT nResponses = (DAAL_INT)rr->getNumberOfResponses();
    DAAL_INT nBetasIntercept = nBetas;
    if (!rr->getInterceptFlag()) { nBetasIntercept--; }

    algorithmFPType *betaBuffer = service_scalable_malloc<algorithmFPType, cpu>(nResponses * nBetas );
    if (!betaBuffer) { _errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Retrieve matrices R and Q'*Y from daal::algorithms::Model */
    NumericTable *rTable, *qtyTable;
    BlockDescriptor<algorithmFPType> rBD, qtyBD;
    algorithmFPType *qrR, *qrQTY;
    getModelPartialSums<algorithmFPType, cpu>(aa, nBetasIntercept, nResponses, readOnly, &rTable, rBD, &qrR, &qtyTable, qtyBD, &qrQTY);

    finalizeQR<algorithmFPType, cpu>(&nBetasIntercept, qrR, &nResponses, qrQTY, betaBuffer, _errors);

    releaseModelQRPartialSums<algorithmFPType, cpu>(rTable, rBD, qtyTable, qtyBD);

    NumericTablePtr betaTable = rr->getBeta();
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
    service_scalable_free<algorithmFPType, cpu>(betaBuffer);
}


template <typename algorithmFPType, CpuType cpu>
services::Status LinearRegressionTrainBatchKernel<algorithmFPType, training::qrDense, cpu>::compute(
            NumericTable *x, NumericTable *y, linear_regression::Model *r,
            const daal::algorithms::Parameter *par)
{
    bool isOnline = false;
    updatePartialModelQR_threaded<algorithmFPType, cpu>(x, y, r, par, isOnline, this->_errors.get());
    if (!this->_errors->isEmpty()) { DAAL_RETURN_STATUS(); }
    finalizeModelQR<algorithmFPType, cpu>(r, r, this->_errors.get());
    DAAL_RETURN_STATUS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status LinearRegressionTrainOnlineKernel<algorithmFPType, training::qrDense, cpu>::compute(
            NumericTable *x, NumericTable *y, linear_regression::Model *r,
            const daal::algorithms::Parameter *par)
{
    bool isOnline = true;
    updatePartialModelQR_threaded<algorithmFPType, cpu>(x, y, r, par, isOnline, this->_errors.get());
    DAAL_RETURN_STATUS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status LinearRegressionTrainOnlineKernel<algorithmFPType, training::qrDense, cpu>::finalizeCompute(
            linear_regression::Model *a, linear_regression::Model *r,
            const daal::algorithms::Parameter *par)
{
    finalizeModelQR<algorithmFPType, cpu>(a, r, this->_errors.get());
    DAAL_RETURN_STATUS();
}

}
}
}
}
}

#endif
