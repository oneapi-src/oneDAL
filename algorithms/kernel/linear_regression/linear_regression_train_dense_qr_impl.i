/* file: linear_regression_train_dense_qr_impl.i */
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
//  Implementation of auxiliary functions for linear regression qrDense method.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_QR_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_QR_IMPL_I__

#include "threading.h"
#include "service_lapack.h"
#include "service_memory.h"
#include "linear_regression_train_kernel.h"

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
template <typename interm, CpuType cpu>
static void getModelPartialSums(ModelQR *model,
                         MKL_INT dim, MKL_INT ny, ReadWriteMode rwmode,
                         NumericTable **rTable,   BlockDescriptor<interm> &rBD,   interm **r,
                         NumericTable **qtyTable, BlockDescriptor<interm> &qtyBD, interm **qty)
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
template <typename interm, CpuType cpu>
static void releaseModelQRPartialSums(NumericTable *rTable,   BlockDescriptor<interm> &rBD,
                                      NumericTable *qtyTable, BlockDescriptor<interm> &qtyBD)
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

template <typename interm, CpuType cpu>
static void computeQRWorkSize(MKL_INT *p, MKL_INT *n, interm *x, interm *tau, MKL_INT *lwork,
            services::KernelErrorCollection *_errors)
{
    MKL_INT info = 0;
    interm work_local;

    *lwork = -1;
    Lapack<interm, cpu>::xxgerqf(p, n, x, p, tau, &work_local, lwork, &info);


    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }

    *lwork = (MKL_INT)work_local;
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
template <typename interm, CpuType cpu>
static void mallocQRWorkBuffer(MKL_INT *p, MKL_INT *n, interm *x, MKL_INT *ny, interm *y, interm *tau,
            interm **work, MKL_INT *lwork, services::KernelErrorCollection *_errors)
{
    MKL_INT info = 0;
    MKL_INT lwork1;

    computeQRWorkSize<interm, cpu>(p, n, x, tau, &lwork1, _errors);
    if (!_errors->isEmpty()) { return; }

    char side = 'R';
    char trans = 'T';
    MKL_INT lwork2 = -1;
    interm work_local;

    Lapack<interm, cpu>::xxormrq(&side, &trans, ny, n, p, x, p, tau, y, ny, &work_local, &lwork2, &info);

    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }

    lwork2 = (MKL_INT)work_local;
    *lwork = ((lwork1 > lwork2) ? lwork1 : lwork2);

    *work = service_scalable_malloc<interm, cpu>((*lwork));
    if (!(*work)) { _errors->add(services::ErrorMemoryAllocationFailed); return; }

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
template <typename interm, CpuType cpu>
static void copyDataToBuffer(MKL_INT *dim, MKL_INT *betaDim, MKL_INT *n, interm *x, MKL_INT *ny, interm *y,
            interm *qrBuffer, interm *qtyBuffer)
{
    MKL_INT iOne = 1;             // integer one
    MKL_INT dimVal = *dim;
    MKL_INT betaDimVal = *betaDim;
    MKL_INT nVal = *n;
    MKL_INT ySize = (*ny) * nVal;

    /* Copy matrix X to temporary buffer in order not to damage it */
    if (dimVal == betaDimVal)
    {
        MKL_INT xSize = dimVal * nVal;
        daal::services::daal_memcpy_s(qrBuffer,xSize*sizeof(interm),x,xSize*sizeof(interm));
    }
    else
    {
        for (size_t i = 0; i < nVal; i++)
        {
            daal::services::daal_memcpy_s(qrBuffer + i * betaDimVal, dimVal * sizeof(interm), x + i * dimVal, dimVal * sizeof(interm));
            qrBuffer[i * betaDimVal + betaDimVal - 1] = 1.0;
        }
    }

    /* Copy matrix Y to temporary buffer in order not to damage it */
    daal::services::daal_memcpy_s(qtyBuffer,ySize*sizeof(interm),y,ySize*sizeof(interm));
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
template <typename interm, CpuType cpu>
static void computeQRForBlock(MKL_INT *p, MKL_INT *n, interm *x, MKL_INT *ny, interm *y, interm *r, interm *qty,
            interm *tau, interm *work, MKL_INT *lwork, services::KernelErrorCollection *_errors)
{
    MKL_INT iOne = 1;             // integer one
    MKL_INT info = 0;
    MKL_INT pVal = *p;
    MKL_INT n_val = *n;
    MKL_INT ny_val = *ny;
    MKL_INT qtySize = ny_val * pVal;
    MKL_INT rOffset = (n_val - pVal) * pVal;
    MKL_INT yqtOffset = (n_val - pVal) * ny_val;

    /* Calculate RQ decomposition of X */
    Lapack<interm, cpu>::xxgerqf(p, n, x, p, tau, work, lwork, &info);

    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }

    /* Copy result into matrix R */
    interm *xPtr = x + rOffset;
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
    Lapack<interm, cpu>::xxormrq(&side, &trans, ny, n, p, x, p, tau, y, ny, work, lwork, &info);
    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }

    /* Copy result into matrix QTY */
    daal::services::daal_memcpy_s(qty, qtySize * sizeof(interm), y + yqtOffset, qtySize * sizeof(interm));
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
template <typename interm, CpuType cpu>
static void mergeQR(MKL_INT *p, MKL_INT *ny, interm *r1,  interm *qty1,  interm *r2, interm *qty2,
            interm *r12, interm *qty12, interm *r,  interm *qty, interm *tau, interm *work, MKL_INT *lwork,
            services::KernelErrorCollection *_errors)
{
    MKL_INT iOne = 1;             // integer one
    MKL_INT p_val = *p;
    MKL_INT n_val = 2 * p_val;
    MKL_INT ny_val = *ny;
    MKL_INT rSize = p_val * p_val;
    MKL_INT qtySize = p_val * ny_val;

    /* Copy R1 and R2 into R12. R12 = (R1, R2) */
    daal::services::daal_memcpy_s(r12        ,rSize*2*sizeof(interm),r1,rSize*sizeof(interm));
    daal::services::daal_memcpy_s(r12 + rSize,rSize*sizeof(interm),  r2,rSize*sizeof(interm));
    /* Copy QTY1 and QTY2 into QTY12. QTY12 = (QTY1, QTY2) */
    daal::services::daal_memcpy_s(qty12          ,qtySize*2*sizeof(interm),qty1,qtySize*sizeof(interm));
    daal::services::daal_memcpy_s(qty12 + qtySize,qtySize*sizeof(interm),  qty2,qtySize*sizeof(interm));

    computeQRForBlock<interm, cpu>(p, &n_val, r12, ny, qty12, r, qty, tau, work, lwork, _errors);
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
template <typename interm, CpuType cpu>
static void finalizeQR(MKL_INT *p, interm *r, MKL_INT *ny, interm *qty, interm *beta,
            services::KernelErrorCollection *_errors)
{
    MKL_INT iOne = 1;             // integer one
    MKL_INT info = 0;
    MKL_INT betaSize = (*ny) * (*p);
    MKL_INT pVal = *p;
    MKL_INT ny_val = *ny;

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
    Lapack<interm, cpu>::xtrtrs(&uplo, &trans, &diag, p, ny, r, p, beta, p, &info);
    if (info != 0) { _errors->add(services::ErrorLinearRegressionInternal); return; }
}

template <typename interm, CpuType cpu>
void updatePartialModelQR(NumericTable *x, NumericTable *y,
            linear_regression::Model *r,
            const daal::algorithms::Parameter *par, bool isOnline,
            services::KernelErrorCollection *_errors)
{
    const linear_regression::Parameter *parameter = static_cast<const linear_regression::Parameter *>(par);
    ModelQR *rr = static_cast<ModelQR *>(r);
    MKL_INT nFeatures = (MKL_INT)x->getNumberOfColumns();
    MKL_INT nResponses = (MKL_INT)y->getNumberOfColumns();
    MKL_INT nRows = (MKL_INT)x->getNumberOfRows();
    MKL_INT nBetas = (MKL_INT)rr->getNumberOfBetas();
    MKL_INT nBetasIntercept = nBetas;
    if (parameter && !parameter->interceptFlag) { nBetasIntercept--; }

    /* Retrieve data associated with input tables */
    BlockDescriptor<interm> xBD;
    BlockDescriptor<interm> yBD;
    x->getBlockOfRows(0, nRows, readOnly, xBD);
    y->getBlockOfRows(0, nRows, readOnly, yBD);
    interm *dy = yBD.getBlockPtr();
    interm *dx = xBD.getBlockPtr();

    /* Retrieve matrices R and Q'*Y from daal::algorithms::Model */
    NumericTable *rTable, *qtyTable;
    BlockDescriptor<interm> rBD, qtyBD;
    interm *qrR, *qrQTY;
    getModelPartialSums<interm, cpu>(rr, nBetasIntercept, nResponses, readWrite, &rTable, rBD, &qrR, &qtyTable, qtyBD, &qrQTY);

    interm *qrROld, *qrQTYOld, *qrRMerge, *qrQTYMerge;
    if (isOnline)
    {
        qrROld     = service_scalable_malloc<interm, cpu>(nBetasIntercept * nBetasIntercept );
        qrQTYOld   = service_scalable_malloc<interm, cpu>(nBetasIntercept * nResponses );
        qrRMerge   = service_scalable_malloc<interm, cpu>(2 * nBetasIntercept * nBetasIntercept );
        qrQTYMerge = service_scalable_malloc<interm, cpu>(2 * nBetasIntercept * nResponses );
        if (!qrROld || !qrQTYOld || !qrRMerge || !qrQTYMerge)
        { _errors->add(services::ErrorMemoryAllocationFailed); return; }

        daal::services::daal_memcpy_s(qrROld, nBetasIntercept * nBetasIntercept * sizeof(interm), qrR,
                                              nBetasIntercept * nBetasIntercept * sizeof(interm));
        daal::services::daal_memcpy_s(qrQTYOld, nBetasIntercept * nResponses * sizeof(interm), qrQTY,
                                                nBetasIntercept * nResponses * sizeof(interm));
    }

    MKL_INT lwork = -1;
    interm *tau       = service_scalable_malloc<interm, cpu>(nBetasIntercept );
    interm *qrBuffer  = service_scalable_malloc<interm, cpu>(nBetasIntercept * nRows );
    interm *qtyBuffer = service_scalable_malloc<interm, cpu>(nResponses * nRows );
    interm *work;
    if (!tau || !qrBuffer || !qtyBuffer)
    { _errors->add(services::ErrorMemoryAllocationFailed); return; }

    mallocQRWorkBuffer<interm, cpu>(&nBetasIntercept, &nRows, dx, &nResponses, dy, tau, &work, &lwork, _errors);
    if(!_errors->isEmpty()) { return; }

    copyDataToBuffer<interm, cpu>(&nFeatures, &nBetasIntercept, &nRows, dx, &nResponses, dy, qrBuffer, qtyBuffer);

    computeQRForBlock<interm, cpu>(&nBetasIntercept, &nRows, qrBuffer, &nResponses, qtyBuffer, qrR, qrQTY,
        tau, work, &lwork, _errors);
    if(!_errors->isEmpty()) { return; }

    x->releaseBlockOfRows(xBD);
    y->releaseBlockOfRows(yBD);

    if (isOnline)
    {
        mergeQR<interm, cpu>(&nBetasIntercept, &nResponses, qrR, qrQTY, qrROld, qrQTYOld,
                qrRMerge, qrQTYMerge, qrR, qrQTY, tau, work, &lwork, _errors);
        if(!_errors->isEmpty()) { return; }
        service_scalable_free<interm, cpu>(qrROld);
        service_scalable_free<interm, cpu>(qrQTYOld);
        service_scalable_free<interm, cpu>(qrRMerge);
        service_scalable_free<interm, cpu>(qrQTYMerge);
    }
    service_scalable_free<interm, cpu>(tau);
    service_scalable_free<interm, cpu>(qrBuffer);
    service_scalable_free<interm, cpu>(qtyBuffer);
    service_scalable_free<interm, cpu>(work);

    releaseModelQRPartialSums<interm, cpu>(rTable, rBD, qtyTable, qtyBD);
}


template<typename interm, CpuType cpu>
struct tls_task_t
{
    interm *work;
    interm *tau;
    interm *qrBuffer;
    interm *qtyBuffer;

    interm *qrR;
    interm *qrQTY;

    interm *qrRNew;
    interm *qrQTYNew;
    interm *qrRMerge;
    interm *qrQTYMerge;

    MKL_INT lwork;
};


template <typename interm, CpuType cpu>
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

    MKL_INT nFeatures = (MKL_INT)x->getNumberOfColumns();
    MKL_INT nResponses = (MKL_INT)y->getNumberOfColumns();
    MKL_INT nRows = (MKL_INT)x->getNumberOfRows();
    MKL_INT nBetas = (MKL_INT)rr->getNumberOfBetas();
    MKL_INT nBetasIntercept = nBetas;
    if (parameter && !parameter->interceptFlag) { nBetasIntercept--; }

    /* Retrieve data associated with input tables */
    BlockDescriptor<interm> xBD;
    BlockDescriptor<interm> yBD;

    x->getBlockOfRows(0, nRows, readOnly, xBD);
    y->getBlockOfRows(0, nRows, readOnly, yBD);

    interm *dy = yBD.getBlockPtr();
    interm *dx = xBD.getBlockPtr();

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
    BlockDescriptor<interm> rBD, qtyBD;
    interm *qrR, *qrQTY;
    getModelPartialSums<interm, cpu>(rr, nBetasIntercept, nResponses, readWrite, &rTable, rBD, &qrR, &qtyTable, qtyBD, &qrQTY);
    if(!isOnline)
    {
        service_memset<interm, cpu>(qrR,   (interm)0, nBetasIntercept * nBetasIntercept);
        service_memset<interm, cpu>(qrQTY, (interm)0, nBetasIntercept * nResponses);
    }

    daal::tls<tls_task_t<interm,cpu>*> *tls_task = new daal::tls<tls_task_t<interm,cpu>*>( [=]()-> tls_task_t<interm,cpu>*
    {
        tls_task_t<interm,cpu>* tt_local = new tls_task_t<interm,cpu>;
        if (!tt_local){ _errors->add(services::ErrorMemoryAllocationFailed); return 0; }

        tt_local->work       = (interm*)0;
        tt_local->lwork      = -1;

        tt_local->tau        = service_scalable_malloc<interm, cpu>(nBetasIntercept);

        tt_local->qrBuffer   = service_scalable_malloc<interm, cpu>(nBetasIntercept * nRowsInLastBlock);
        tt_local->qtyBuffer  = service_scalable_malloc<interm, cpu>(nResponses * nRowsInLastBlock);

        tt_local->qrR        = service_scalable_calloc<interm, cpu>(nBetasIntercept * nBetasIntercept);
        tt_local->qrQTY      = service_scalable_calloc<interm, cpu>(nBetasIntercept * nResponses);

        tt_local->qrRNew     = service_scalable_calloc<interm, cpu>(nBetasIntercept * nBetasIntercept);
        tt_local->qrQTYNew   = service_scalable_calloc<interm, cpu>(nBetasIntercept * nResponses);

        tt_local->qrRMerge   = service_scalable_malloc<interm, cpu>(2 * nBetasIntercept * nBetasIntercept);
        tt_local->qrQTYMerge = service_scalable_malloc<interm, cpu>(2 * nBetasIntercept * nResponses);

        interm* dx_local = dx;
        interm* dy_local = dy;

        MKL_INT nBetasIntercept_local = nBetasIntercept;
        MKL_INT nResponses_local      = nResponses;
        MKL_INT nRowsInLastBlock_local       = nRowsInLastBlock;

        services::KernelErrorCollection *_errors_local = _errors;


        // Function that allocates memory for storing intermediate data for qrDense decomposition
        mallocQRWorkBuffer<interm, cpu>(&nBetasIntercept_local,  // in      Number of columns in input matrix X
                                        &nRowsInLastBlock_local, // in      Number of rows in input matrix X
                                        dx_local,                // in      Input matrix X of size (n x p), n > p
                                        &nResponses_local,       // in      Number of columns in input matrix Y
                                        dy_local,                // in      Input matrix Y of size (n x ny)
                                        tt_local->tau,           // in      LAPACK GERQF/ORMRQ TAU parameter. Array of size p
                                        &(tt_local->work),       // out     LAPACK GERQF/ORMRQ WORK parameter
                                        &(tt_local->lwork),      // out     Calculated size of WORK array
                                        _errors_local); if(!_errors_local->isEmpty()) { return 0; }

        if (!tt_local->qrRNew || !tt_local->qrQTYNew || !tt_local->qrRMerge || !tt_local->qrQTYMerge || !tt_local->tau || !tt_local->qrBuffer || !tt_local->qtyBuffer){ _errors->add(services::ErrorMemoryAllocationFailed); return 0; }
        return tt_local;

    } ); /* Allocate memory for all arrays inside TLS: end */


    daal::threader_for( nBlocks, nBlocks, [=, &tls_task](int iBlock)
    {
        struct tls_task_t<interm,cpu> * tt_local = tls_task->local();

        interm** pwork_local     = &(tt_local->work);
        MKL_INT* plwork_local    = &(tt_local->lwork);

        interm* tau_local        = tt_local->tau;
        interm* qrBuffer_local   = tt_local->qrBuffer;
        interm* qtyBuffer_local  = tt_local->qtyBuffer;

        interm* qrR_local        = tt_local->qrR;
        interm* qrQTY_local      = tt_local->qrQTY;

        interm* qrRNew_local     = tt_local->qrRNew;
        interm* qrQTYNew_local   = tt_local->qrQTYNew;

        interm* qrRMerge_local   = tt_local->qrRMerge;
        interm* qrQTYMerge_local = tt_local->qrQTYMerge;

        MKL_INT nBetasIntercept_local = nBetasIntercept;
        MKL_INT nResponses_local      = nResponses;
        MKL_INT nFeatures_local       = nFeatures;

        services::KernelErrorCollection *_errors_local = _errors;

        size_t startRow = iBlock * nRowsInBlock;
        size_t nCurrentRowsInBlock = (iBlock < (nBlocks-1))?nRowsInBlock:nRowsInLastBlock;
        size_t endRow = startRow + nCurrentRowsInBlock;
        MKL_INT nN_local = endRow - startRow;

        interm* dx_local = dx + startRow * nFeatures;
        interm* dy_local = dy + startRow * nResponses;


        // Function that copies input matrices X and Y into intermediate buffers.
        copyDataToBuffer<interm, cpu>(  &nFeatures_local,        // in      Number of columns in input matrix X
                                        &nBetasIntercept_local,  // in      Number of regression coefficients
                                        &nN_local,               // in      Number of rows in input matrix X
                                        dx_local,                // in      Input matrix X of size (n x p), n > p
                                        &nResponses_local,       // in      Number of columns in input matrix Y
                                        dy_local,                // in      Input matrix Y of size (n x ny)
                                        qrBuffer_local,          // out     if (dim == betaDim) copy of matrix X, if (dim + 1 == betaDim) qrBuffer = (X|e) where e is a column vector of all 1's.
                                        qtyBuffer_local);        // out     copy of matrix Y

        // Function that calculates R and Y*Q' from input matrix  of independent variables X and matrix of responses Y.
        computeQRForBlock<interm, cpu>( &nBetasIntercept_local,  // in      Number of columns in input matrix X
                                        &nN_local,               // in      Number of rows in input matrix X
                                        qrBuffer_local,          // in, out Input matrix X of size (n x p), n > p; Overwritten by LAPACK on output
                                        &nResponses_local,       // in      Number of columns in input matrix Y
                                        qtyBuffer_local,         // in, out Input matrix Y of size (n x ny); Overwritten by LAPACK on output
                                        qrRNew_local,            // out     Matrix R of size (p x p)
                                        qrQTYNew_local,          // out     Matrix Y*Q' of size (ny x p)
                                        tau_local,               // in      LAPACK GERQF/ORMRQ TAU parameter. Array of size p
                                        *pwork_local,            // in      LAPACK GERQF/ORMRQ WORK parameter
                                        plwork_local,            // in      Calculated size of WORK array
                                        _errors_local); if(!_errors_local->isEmpty()) { return; }

        // Function that merges qrDense partial sums (R1, QTY1), (R2, QTY2) into partial sum (R, QTY)
        mergeQR<interm, cpu>(           &nBetasIntercept_local,  // in      Number of rows and columns in R, R1, R2
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
                                        *pwork_local,            // in      LAPACK GERQF WORK parameter
                                        plwork_local,            // in      Size of WORK array
                                        _errors_local); if(!_errors_local->isEmpty()) { return; }

    } ); /* for(iBlock = 0; iBlock < nBlocks; iBlock++) */


    tls_task->reduce( [=](tls_task_t<interm,cpu>* tt_local)-> void
    {
        interm** pwork_local     = &(tt_local->work);
        MKL_INT* plwork_local    = &(tt_local->lwork);

        interm* tau_local        = tt_local->tau;

        interm* qrBuffer_local   = tt_local->qrBuffer;
        interm* qtyBuffer_local  = tt_local->qtyBuffer;

        interm* qrR_local     = tt_local->qrR;
        interm* qrQTY_local   = tt_local->qrQTY;

        interm* qrRNew_local     = tt_local->qrRNew;
        interm* qrQTYNew_local   = tt_local->qrQTYNew;

        interm* qrRMerge_local   = tt_local->qrRMerge;
        interm* qrQTYMerge_local = tt_local->qrQTYMerge;

        MKL_INT nBetasIntercept_local = nBetasIntercept;
        MKL_INT nResponses_local      = nResponses;
        MKL_INT nFeatures_local       = nFeatures;
        services::KernelErrorCollection *_errors_local = _errors;

        // Function that merges qrDense partial sums (R1, QTY1), (R2, QTY2) into partial sum (R, QTY)
        mergeQR<interm, cpu>(           &nBetasIntercept_local,  // in      Number of rows and columns in R, R1, R2
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
                                        *pwork_local,            // in      LAPACK GERQF WORK parameter
                                        plwork_local,            // in      Size of WORK array
                                        _errors_local); if(!_errors_local->isEmpty()) { return; }

        service_scalable_free<interm, cpu>(tt_local->work); tt_local->work = 0;

        service_scalable_free<interm, cpu>(tt_local->qrRNew);
        service_scalable_free<interm, cpu>(tt_local->qrQTYNew);

        service_scalable_free<interm, cpu>(tt_local->qrR);
        service_scalable_free<interm, cpu>(tt_local->qrQTY);

        service_scalable_free<interm, cpu>(tt_local->qrRMerge);
        service_scalable_free<interm, cpu>(tt_local->qrQTYMerge);

        service_scalable_free<interm, cpu>(tt_local->qrBuffer);
        service_scalable_free<interm, cpu>(tt_local->qtyBuffer);

        service_scalable_free<interm, cpu>(tt_local->tau);

        delete tt_local;
    } );
    delete tls_task;

    x->releaseBlockOfRows(xBD);
    y->releaseBlockOfRows(yBD);

    releaseModelQRPartialSums<interm, cpu>(rTable, rBD, qtyTable, qtyBD);
}




template <typename interm, CpuType cpu>
void finalizeModelQR(linear_regression::Model *a, linear_regression::Model *r,
                   services::KernelErrorCollection *_errors)
{
    ModelQR *aa = static_cast<ModelQR *>(a);
    ModelQR *rr = static_cast<ModelQR *>(r);

    MKL_INT nBetas = (MKL_INT)rr->getNumberOfBetas();
    MKL_INT nResponses = (MKL_INT)rr->getNumberOfResponses();
    MKL_INT nBetasIntercept = nBetas;
    if (!rr->getInterceptFlag()) { nBetasIntercept--; }

    interm *betaBuffer = service_scalable_malloc<interm, cpu>(nResponses * nBetas );
    if (!betaBuffer) { _errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Retrieve matrices R and Q'*Y from daal::algorithms::Model */
    NumericTable *rTable, *qtyTable;
    BlockDescriptor<interm> rBD, qtyBD;
    interm *qrR, *qrQTY;
    getModelPartialSums<interm, cpu>(aa, nBetasIntercept, nResponses, readOnly, &rTable, rBD, &qrR, &qtyTable, qtyBD, &qrQTY);

    finalizeQR<interm, cpu>(&nBetasIntercept, qrR, &nResponses, qrQTY, betaBuffer, _errors);

    releaseModelQRPartialSums<interm, cpu>(rTable, rBD, qtyTable, qtyBD);

    NumericTablePtr betaTable = rr->getBeta();
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
    service_scalable_free<interm, cpu>(betaBuffer);
}


template <typename interm, CpuType cpu>
void LinearRegressionTrainBatchKernel<interm, training::qrDense, cpu>::compute(
            NumericTable *x, NumericTable *y, linear_regression::Model *r,
            const daal::algorithms::Parameter *par)
{
    bool isOnline = false;
    updatePartialModelQR_threaded<interm, cpu>(x, y, r, par, isOnline, this->_errors.get());
    if (!this->_errors->isEmpty()) { return; }
    finalizeModelQR<interm, cpu>(r, r, this->_errors.get());
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainOnlineKernel<interm, training::qrDense, cpu>::compute(
            NumericTable *x, NumericTable *y, linear_regression::Model *r,
            const daal::algorithms::Parameter *par)
{
    bool isOnline = true;
    updatePartialModelQR_threaded<interm, cpu>(x, y, r, par, isOnline, this->_errors.get());
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainOnlineKernel<interm, training::qrDense, cpu>::finalizeCompute(
            linear_regression::Model *a, linear_regression::Model *r,
            const daal::algorithms::Parameter *par)
{
    finalizeModelQR<interm, cpu>(a, r, this->_errors.get());
}

}
}
}
}
}

#endif
