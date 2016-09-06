/* file: implicit_als_train_dense_default_batch_impl.i */
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
//  Implementation of impicit ALS training algorithm for batch processing mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_IMPL_I__

#include "threading.h"
#include "service_blas.h"
#include "service_lapack.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernelCommon<algorithmFPType, cpu>::computeXtX(
    size_t *nRows, size_t *nCols, algorithmFPType *beta, algorithmFPType *x, size_t *ldx,
    algorithmFPType *xtx, size_t *ldxtx)
{
    /* SYRK parameters */
    char uplo = 'U';
    char trans = 'N';
    algorithmFPType alpha = 1.0;

    Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, (MKL_INT *)nCols, (MKL_INT *)nRows, &alpha, x, (MKL_INT *)ldx, beta,
                       xtx, (MKL_INT *)ldxtx);
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernelBase<algorithmFPType, cpu>::updateSystem(
    size_t *nCols, algorithmFPType *x, algorithmFPType *coeff, algorithmFPType *c,
    algorithmFPType *a, size_t *lda, algorithmFPType *b)
{
    /* SYR parameters */
    char uplo = 'U';
    MKL_INT iOne = 1;
    Blas<algorithmFPType, cpu>::xsyr(&uplo, (MKL_INT *)nCols, coeff, x, &iOne, a, (MKL_INT *)lda);

    if (*coeff > 0.0)
    {
        Blas<algorithmFPType, cpu>::xaxpy((MKL_INT *)nCols, c, x, &iOne, b, &iOne);
    }
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernelBase<algorithmFPType, cpu>::solve(
    size_t *nCols, algorithmFPType *a, size_t *lda,
    algorithmFPType *b, size_t *ldb)
{
    /* POTRF parameters */
    char uplo = 'U';
    MKL_INT iOne = 1;
    MKL_INT info = 0;

    /* Perform L*L' decomposition of A */
    Lapack<algorithmFPType, cpu>::xpotrf(&uplo, (MKL_INT *)nCols, a, (MKL_INT *)lda, &info);
    if ( info != 0 ) { this->_errors->add(services::ErrorALSInternal); return; }

    /* Solve L*L' * x = b */
    Lapack<algorithmFPType, cpu>::xpotrs(&uplo, (MKL_INT *)nCols, &iOne, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, &info);
    if ( info != 0 ) { this->_errors->add(services::ErrorALSInternal); return; }
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernelBase<algorithmFPType, cpu>::computeFactors(
    size_t nRows, size_t nCols, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
    size_t nFactors, algorithmFPType *colFactors, algorithmFPType *rowFactors,
    algorithmFPType alpha, algorithmFPType lambda, algorithmFPType *xtx, daal::tls<algorithmFPType *> *lhs)
{
    daal::threader_for(nRows, nRows, [ & ](size_t i)
    {
        algorithmFPType *lhs_local = lhs->local();
        algorithmFPType *rhs = rowFactors + i * nFactors;
        service_memset<algorithmFPType, cpu>(rhs, 0.0, nFactors);
        daal::services::daal_memcpy_s(lhs_local, nFactors * nFactors * sizeof(algorithmFPType),
                                      xtx, nFactors * nFactors * sizeof(algorithmFPType));

        formSystem(i, nCols, data, colIndices, rowOffsets, nFactors, colFactors, alpha, lhs_local, rhs, lambda);

        /* Solve system of normal equations */
        solve(&nFactors, lhs_local, &nFactors, rhs, &nFactors);
        if (!this->_errors->isEmpty())
        { return; }
    } );
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu>::computeCostFunction(
    size_t nUsers, size_t nItems, size_t nFactors, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
    algorithmFPType *itemsFactors, algorithmFPType *usersFactors, algorithmFPType alpha, algorithmFPType lambda,
    algorithmFPType *costFunctionPtr)
{
    algorithmFPType zero = 0.0;
    algorithmFPType one = 1.0;
    algorithmFPType costFunction = zero;
    algorithmFPType sumUsers2 = zero;
    algorithmFPType sumItems2 = zero;

    for (size_t i = 0; i < nUsers; i++)
    {
        size_t startIdx = rowOffsets[i]   - 1;
        size_t endIdx   = rowOffsets[i + 1] - 1;
        algorithmFPType *usersI = usersFactors + i * nFactors;
        for (size_t j = startIdx; j < endIdx; j++)
        {
            algorithmFPType c = one + alpha * data[j];
            algorithmFPType *itemsJ = itemsFactors + (colIndices[j] - 1) * nFactors;

            algorithmFPType dotProduct = 0.0;
            for (size_t k = 0; k < nFactors; k++)
            {
                dotProduct += usersI[k] * itemsJ[k];
            }
            algorithmFPType sqrError = 1.0 - dotProduct;
            sqrError *= sqrError;
            costFunction += c * sqrError;
        }
    }
    for (size_t i = 0; i < nItems * nFactors; i++)
    {
        sumItems2 += itemsFactors[i] * itemsFactors[i];
    }
    for (size_t i = 0; i < nUsers * nFactors; i++)
    {
        sumUsers2 += usersFactors[i] * usersFactors[i];
    }
    costFunction += lambda * (sumItems2 + sumUsers2);
    *costFunctionPtr = costFunction;
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernel<algorithmFPType, defaultDense, cpu>::computeCostFunction(
    size_t nUsers, size_t nItems, size_t nFactors, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
    algorithmFPType *itemsFactors, algorithmFPType *usersFactors, algorithmFPType alpha, algorithmFPType lambda,
    algorithmFPType *costFunctionPtr)
{
    algorithmFPType zero = 0.0;
    algorithmFPType one = 1.0;
    algorithmFPType costFunction = zero;
    algorithmFPType sumUsers2 = zero;
    algorithmFPType sumItems2 = zero;

    for (size_t i = 0; i < nUsers; i++)
    {
        algorithmFPType *usersI = usersFactors + i * nFactors;
        for (size_t j = 0; j < nItems; j++)
        {
            if (data[i * nItems + j] > 0.0)
            {
                algorithmFPType *itemsJ = itemsFactors + j * nFactors;
                algorithmFPType c = one + alpha * data[i * nItems + j];

                algorithmFPType dotProduct = 0.0;
                for (size_t k = 0; k < nFactors; k++)
                {
                    dotProduct += usersI[k] * itemsJ[k];
                }
                algorithmFPType sqrError = one - dotProduct;
                sqrError *= sqrError;
                costFunction += c * sqrError;
            }
        }
    }
    for (size_t i = 0; i < nItems * nFactors; i++)
    {
        sumItems2 += itemsFactors[i] * itemsFactors[i];
    }
    for (size_t i = 0; i < nUsers * nFactors; i++)
    {
        sumUsers2 += usersFactors[i] * usersFactors[i];
    }
    costFunction += lambda * (sumItems2 + sumUsers2);
    *costFunctionPtr = costFunction;
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu>::formSystem(
    size_t i, size_t nCols, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
    size_t nFactors, algorithmFPType *colFactors,
    algorithmFPType alpha, algorithmFPType *lhs, algorithmFPType *rhs, algorithmFPType lambda)
{
    size_t startIdx = rowOffsets[i]   - 1;
    size_t endIdx   = rowOffsets[i + 1] - 1;
    /* Update the linear system of normal equations */
    for (size_t j = startIdx; j < endIdx; j++)
    {
        algorithmFPType c1 = alpha * data[j];
        algorithmFPType c = c1 + 1.0;
        algorithmFPType *colFactorsRow = colFactors + (colIndices[j] - 1) * nFactors;

        this->updateSystem(&nFactors, colFactorsRow, &c1, &c, lhs, &nFactors, rhs);
    }

    /* Add regularization term */
    algorithmFPType gamma = lambda * (endIdx - startIdx);
    for (size_t k = 0; k < nFactors; k++)
    {
        lhs[k * nFactors + k] += gamma;
    }
}


template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernel<algorithmFPType, defaultDense, cpu>::formSystem(
    size_t i, size_t nCols, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
    size_t nFactors, algorithmFPType *colFactors,
    algorithmFPType alpha, algorithmFPType *lhs, algorithmFPType *rhs, algorithmFPType lambda)
{
    algorithmFPType one = 1.0;
    algorithmFPType gammaMultiplier = 1.0;
    /* Update the linear system of normal equations */
    for (size_t j = 0; j < nCols; j++)
    {
        algorithmFPType rating = data[i * nCols + j];
        if (rating > 0.0)
        {
            algorithmFPType c1 = alpha * rating;
            algorithmFPType c = c1 + 1.0;
            algorithmFPType *colFactorsRow = colFactors + j * nFactors;

            this->updateSystem(&nFactors, colFactorsRow, &c1, &c, lhs, &nFactors, rhs);
            gammaMultiplier += one;
        }
    }

    /* Add regularization term */
    algorithmFPType gamma = lambda * gammaMultiplier;
    for (size_t k = 0; k < nFactors; k++)
    {
        lhs[k * nFactors + k] += gamma;
    }
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainBatchKernel<algorithmFPType, fastCSR, cpu>::compute(const NumericTable *dataTable,
                                                                         implicit_als::Model *initModel,
                                                                         implicit_als::Model *model,
                                                                         const Parameter *parameter)
{
    ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu> task(dataTable, initModel, model, parameter, this);
    if (!this->_errors->isEmpty()) { return; }

    size_t maxIterations   = parameter->maxIterations;
    algorithmFPType alpha  = (algorithmFPType)(parameter->alpha);
    algorithmFPType lambda = (algorithmFPType)(parameter->lambda);

    size_t nItems = task.nItems;
    size_t nUsers = task.nUsers;
    size_t nFactors = task.nFactors;
    algorithmFPType *itemsFactors = task.itemsFactors;
    algorithmFPType *usersFactors = task.usersFactors;
    algorithmFPType *xtx = task.xtx;
    daal::tls<algorithmFPType *> *lhs = task.lhs;

    algorithmFPType *data = task.data;
    algorithmFPType *tdata = task.tdata;
    size_t *colIndices = task.colIndices;
    size_t *rowOffsets = task.rowOffsets;
    size_t *rowIndices = task.rowIndices;
    size_t *colOffsets = task.colOffsets;

#if 0
    algorithmFPType costFunction;
    computeCostFunction(nUsers, nItems, nFactors, data, colIndices, rowOffsets, itemsFactors, usersFactors,
                        alpha, lambda, &costFunction);
#endif
    algorithmFPType beta = 0.0;
    for (size_t i = 0; i < maxIterations; i++)
    {
        this->computeXtX(&nItems, &nFactors, &beta, itemsFactors, &nFactors, xtx, &nFactors);

        this->computeFactors(nUsers, nItems, data, colIndices, rowOffsets, nFactors, itemsFactors, usersFactors,
                             alpha, lambda, xtx, lhs);
        if (!this->_errors->isEmpty()) { return; }

        this->computeXtX(&nUsers, &nFactors, &beta, usersFactors, &nFactors, xtx, &nFactors);

        this->computeFactors(nItems, nUsers, tdata, rowIndices, colOffsets, nFactors, usersFactors, itemsFactors,
                             alpha, lambda, xtx, lhs);
        if (!this->_errors->isEmpty()) { return; }

#if 0
        computeCostFunction(nUsers, nItems, nFactors, data, colIndices, rowOffsets, itemsFactors, usersFactors,
                            alpha, lambda, &costFunction);
#endif
    }
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainBatchKernel<algorithmFPType, defaultDense, cpu>::compute(const NumericTable *dataTable,
                                                                              implicit_als::Model *initModel,
                                                                              implicit_als::Model *model,
                                                                              const Parameter *parameter)
{
    ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu> task(dataTable, initModel, model, parameter, this);
    if (!this->_errors->isEmpty()) { return; }

    size_t maxIterations   = parameter->maxIterations;
    algorithmFPType alpha  = (algorithmFPType)(parameter->alpha);
    algorithmFPType lambda = (algorithmFPType)(parameter->lambda);

    size_t nItems = task.nItems;
    size_t nUsers = task.nUsers;
    size_t nFactors = task.nFactors;
    algorithmFPType *itemsFactors = task.itemsFactors;
    algorithmFPType *usersFactors = task.usersFactors;
    algorithmFPType *xtx = task.xtx;
    daal::tls<algorithmFPType *> *lhs = task.lhs;

    algorithmFPType *data = task.data;
    algorithmFPType *tdata = task.tdata;

#if 0
    algorithmFPType costFunction;
    computeCostFunction(nUsers, nItems, nFactors, data, NULL, NULL, itemsFactors, usersFactors,
                        alpha, lambda, &costFunction);
#endif

    algorithmFPType beta = 0.0;
    for (size_t i = 0; i < maxIterations; i++)
    {
        this->computeXtX(&nItems, &nFactors, &beta, itemsFactors, &nFactors, xtx, &nFactors);

        this->computeFactors(nUsers, nItems, data, NULL, NULL, nFactors, itemsFactors, usersFactors,
                             alpha, lambda, xtx, lhs);
        if (!this->_errors->isEmpty()) { return; }

        this->computeXtX(&nUsers, &nFactors, &beta, usersFactors, &nFactors, xtx, &nFactors);

        this->computeFactors(nItems, nUsers, tdata, NULL, NULL, nFactors, usersFactors, itemsFactors,
                             alpha, lambda, xtx, lhs);
        if (!this->_errors->isEmpty()) { return; }

#if 0
        computeCostFunction(nUsers, nItems, nFactors, data, NULL, NULL, itemsFactors, usersFactors,
                            alpha, lambda, &costFunction);
#endif
    }
}

}
}
}
}
}

#endif
