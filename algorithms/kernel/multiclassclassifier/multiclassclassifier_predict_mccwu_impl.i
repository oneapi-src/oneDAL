/* file: multiclassclassifier_predict_mccwu_impl.i */
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
//  Implementation of Wu method for Multi-class classifier
//  prediction algorithm.
//--
*/
/*
//  REFERENCES
//
//  1. Ting-Fan Wu, Chih-Jen Lin, Ruby C. Weng
//     Probability Estimates for Multi-class Classification by Pairwise Coupling,
//     Journal of Machine Learning Research 5, 2004.
*/

#ifndef __MULTICLASSCLASSIFIER_PREDICT_MCCWU_IMPL_I__
#define __MULTICLASSCLASSIFIER_PREDICT_MCCWU_IMPL_I__

#include "multi_class_classifier_model.h"

#include "threading.h"
#include "service_math.h"
#include "service_memory.h"
#include "service_data_utils.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "service_blas.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
void MultiClassClassifierPredictKernel<multiClassClassifierWu, training::oneAgainstOne, algorithmFPType, cpu>::
    compute(const NumericTable *a, const daal::algorithms::Model *m, NumericTable *r,
            const daal::algorithms::Parameter *par)
{
    Model *model = static_cast<Model *>(const_cast<daal::algorithms::Model *>(m));
    Parameter *mccPar = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

    size_t nClasses = mccPar->nClasses;
    size_t nIter = mccPar->maxIterations;
    double eps = mccPar->accuracyThreshold;

    size_t nFeatures = a->getNumberOfColumns();
    size_t nVectors  = a->getNumberOfRows();

    size_t nRowsInBlock = getMultiClassClassifierPredictBlockSize<algorithmFPType, cpu>();
    /* Calculate number of blocks of rows including tail block */
    size_t nBlocks = nVectors / nRowsInBlock;
    if (nBlocks * nRowsInBlock < nVectors) { nBlocks++; }

    /* Allocate data for storing intermediate results */

    /* Allocate thread local storage */
    daal::tls<MultiClassClassifierTls<algorithmFPType, cpu> *> tls([=]()
    {
        return new MultiClassClassifierTls<algorithmFPType, cpu>(
                nClasses, nRowsInBlock, a, r, mccPar->prediction);
    } );

    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock)
    {
        size_t startRow = iBlock * nRowsInBlock;
        size_t nRows = nRowsInBlock;
        if (startRow + nRows > nVectors) { nRows = nVectors - startRow; }

        MultiClassClassifierTls<algorithmFPType, cpu> *localValues = tls.local();
        MicroTable *mtX = localValues->mtX;
        FeatureMicroTable <int, writeOnly, cpu> &mtR = localValues->mtR;
        services::SharedPtr<classifier::prediction::Batch> simplePrediction = localValues->simplePrediction;
        algorithmFPType *buffer = localValues->buffer;
        services::Error &localError = localValues->error;
        int oldNumberOfThreads = fpk_serv_set_num_threads_local(1);
        getBlockOfRowsOfResults(nFeatures, startRow, nRows, nClasses, mtX, mtR,
                                simplePrediction, model,
                                nIter, eps, buffer, localError);
        fpk_serv_set_num_threads_local(oldNumberOfThreads);
        if(localError.id() != services::NoErrorMessageFound) { return; }
    } );

    tls.reduce([=](MultiClassClassifierTls<algorithmFPType, cpu> *localValues)
    {
        if(localValues->error.id() != services::NoErrorMessageFound)
        {
            this->_errors->add(services::SharedPtr<services::Error>(new services::Error(localValues->error)));
        }
        delete localValues;
    } );
}

template<typename algorithmFPType, CpuType cpu>
inline void MultiClassClassifierPredictKernel<multiClassClassifierWu, training::oneAgainstOne, algorithmFPType, cpu>::
getBlockOfRowsOfResults(size_t nFeatures, size_t startRow, size_t nRows, size_t nClasses,
                        MicroTable *mtX, FeatureMicroTable<int, writeOnly, cpu> &mtR,
                        services::SharedPtr<classifier::prediction::Batch> simplePrediction,
                        Model *model,
                        size_t nIter, double eps, algorithmFPType *buffer,
                        services::Error &error)
{
    algorithmFPType one = 1.0;
    algorithmFPType invNClasses = one / (algorithmFPType)nClasses;

    algorithmFPType *rProb = buffer;
    algorithmFPType *Q     = rProb + nRows * nClasses * nClasses;
    algorithmFPType *Qp    = Q     + nClasses * nClasses;
    algorithmFPType *p     = Qp    + nClasses;
    algorithmFPType *y     = p     + nClasses;

    /* Get 2-class probabilities */
    get2ClassProbabilities(nFeatures, startRow, nRows, nClasses, mtX, y,
                           simplePrediction, model, rProb, error);
    if(error.id() != services::NoErrorMessageFound) { return; }

    algorithmFPType *rProbPtr = rProb;
    size_t nClassesSq = nClasses * nClasses;
    for (size_t k = 0; k < nRows; k++, rProbPtr += nClassesSq)
    {
        /* Set initial probabilities */
        for (size_t j = 0; j < nClasses; j++)
        {
            p[j] = invNClasses;
        }

        /* Calculate matrix Q */
        computeQ(nClasses, rProbPtr, Q);

        algorithmFPType objFuncPrev;
        objFuncPrev = daal::DataFeatureUtils::internal::MaxVal<algorithmFPType, cpu>::get();
        for (size_t it = 0; it < nIter; it++)
        {
            /* Check convergence criteria */
            algorithmFPType objFunc = computeObjFunc(nClasses, p, rProbPtr);
            if (daal::internal::Math<algorithmFPType, cpu>::sFabs(objFunc - objFuncPrev) < eps) { break; }
            objFuncPrev = objFunc;

            /* Update multiclass probabilities estimates */
            updateProbabilities(nClasses, Q, Qp, p);
        }

        /* Calculate resulting classes labels */
        int *label;
        algorithmFPType maxProb = p[0];
        mtR.getBlockOfColumnValues(0, startRow + k, 1, &label);
        label[0] = 0;
        for (int j = 1; j < nClasses; j++)
        {
            if (p[j] > maxProb)
            {
                maxProb = p[j];
                label[0] = j;
            }
        }
        mtR.release();
    }
}

template<typename algorithmFPType, CpuType cpu>
inline void MultiClassClassifierPredictKernel<multiClassClassifierWu, training::oneAgainstOne, algorithmFPType, cpu>::
get2ClassProbabilities(size_t nFeatures, size_t startRow, size_t nRows, size_t nClasses,
                       MicroTable *mtX, algorithmFPType *y,
                       services::SharedPtr<classifier::prediction::Batch> simplePrediction,
                       Model *model,
                       algorithmFPType *rProb, services::Error &error)
{
    algorithmFPType one = 1.0;
    NumericTablePtr xTable;
    if (mtX->getDataLayout() == NumericTableIface::csrArray)
    {
        algorithmFPType *x;
        size_t *colIndices, *rowOffsets;
        static_cast<CSRBlockMicroTable<algorithmFPType, readOnly, cpu> *>(mtX)->getSparseBlock(
                startRow, nRows, &x, &colIndices, &rowOffsets);
        xTable = NumericTablePtr(new CSRNumericTable(x, colIndices, rowOffsets, nFeatures, nRows));
    }
    else
    {
        algorithmFPType *x;
        static_cast<BlockMicroTable<algorithmFPType, readOnly, cpu> *>(mtX)->getBlockOfRows(
                startRow, nRows, &x);
        xTable = NumericTablePtr(new HomogenNumericTableCPU<algorithmFPType, cpu> (x, nFeatures, nRows));
    }
    NumericTablePtr yTable (new HomogenNumericTableCPU<algorithmFPType, cpu> (y, 1, nRows));

    services::SharedPtr<classifier::prediction::Result> yRes(new classifier::prediction::Result());
    if (!xTable || !yTable || !yRes) { error.setId(services::ErrorMemoryAllocationFailed); return; }
    yRes->set(classifier::prediction::prediction, yTable);
    for (size_t i = 1, imodel = 0; i < nClasses; i++)
    {
        for (size_t j = 0; j < i; j++, imodel++)
        {
            /* Compute prediction of the "simple" classifier for pair of labels (i, j) */

            simplePrediction->inputBase->set(classifier::prediction::data, xTable);
            simplePrediction->inputBase->set(classifier::prediction::model, model->getTwoClassClassifierModel(imodel) );

            simplePrediction->setResult(yRes);
            simplePrediction->computeNoThrow();
            if(simplePrediction->getErrors()->size() != 0)
            { error.setId(services::ErrorMultiClassFailedToComputeTwoClassPrediction); return; }

            /* Use sigmoid to calculate probabilities */
            daal::internal::Math<algorithmFPType, cpu>::vExp(nRows, y, y);
            algorithmFPType *rProbPtr = rProb;
            for (size_t k = 0; k < nRows; k++, rProbPtr += nClasses * nClasses)
            {
                algorithmFPType p = one / (one + y[k]);
                rProbPtr[i * nClasses + j] = one - p;
                rProbPtr[j * nClasses + i] = p;
            }
        }
    }
    if (mtX->getDataLayout() == NumericTableIface::csrArray)
    {
        static_cast<CSRBlockMicroTable<algorithmFPType, readOnly, cpu> *>(mtX)->release();
    }
    else
    {
        static_cast<BlockMicroTable<algorithmFPType, readOnly, cpu> *>(mtX)->release();
    }
}

template<typename algorithmFPType, CpuType cpu>
inline void MultiClassClassifierPredictKernel<multiClassClassifierWu, training::oneAgainstOne, algorithmFPType, cpu>::
    computeQ(size_t nClasses, const algorithmFPType *rProb, algorithmFPType *Q)
{
    algorithmFPType zero = 0.0;
    for (size_t i = 0; i < nClasses; i++)
    {
        Q[i * nClasses + i] = zero;
        for (size_t j = 0; j < i; j++)
        {
            algorithmFPType rProbJI = rProb[j * nClasses + i];
            Q[i * nClasses + i] +=  rProbJI * rProbJI;
            Q[i * nClasses + j]  = -rProb[i * nClasses + j] * rProbJI;
            Q[j * nClasses + i]  = Q[i * nClasses + j];
        }
        for (size_t j = i + 1; j < nClasses; j++)
        {
            algorithmFPType rProbJI = rProb[j * nClasses + i];
            Q[i * nClasses + i] +=  rProbJI * rProbJI;
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
inline algorithmFPType MultiClassClassifierPredictKernel<multiClassClassifierWu, training::oneAgainstOne, algorithmFPType, cpu>::
    computeObjFunc(size_t nClasses, algorithmFPType *p, algorithmFPType *rProb)
{
    algorithmFPType objFunc = 0.0;
    for (size_t i = 0; i < nClasses; i++)
    {
        algorithmFPType diff;
        for (size_t j = 0; j < i; j++)
        {
            diff = rProb[i * nClasses + j] * p[i] + rProb[j * nClasses + i] * p[j];
            objFunc += diff * diff;
        }
        for (size_t j = i + 1; j < nClasses; j++)
        {
            diff = rProb[i * nClasses + j] * p[i] + rProb[j * nClasses + i] * p[j];
            objFunc += diff * diff;
        }
    }
    return objFunc;
}

template<typename algorithmFPType, CpuType cpu>
inline void MultiClassClassifierPredictKernel<multiClassClassifierWu, training::oneAgainstOne, algorithmFPType, cpu>::
    updateProbabilities(size_t nClasses, const algorithmFPType *Q,
                        algorithmFPType *Qp, algorithmFPType *p)
{
    algorithmFPType zero = 0.0;
    algorithmFPType one  = 1.0;

    /* Calculate Q*p */
    for (size_t i = 0; i < nClasses; i++)
    {
        Qp[i] = zero;
        for (size_t j = 0; j < nClasses; j++)
        {
            Qp[i] += Q[i * nClasses + j] * p[j];
        }
    }

    /* Calculate p'*Q*p */
    algorithmFPType pQp = zero;
    for (size_t j = 0; j < nClasses; j++)
    {
        pQp += p[j] * Qp[j];
    }

    /* Update probabilities p */
    algorithmFPType sumP = zero;
    for (size_t j = 0; j < nClasses; j++)
    {
        p[j] = (pQp - Qp[j] + Q[j * nClasses + j] * p[j]) / Q[j * nClasses + j];
        sumP += p[j];
    }

    /* Normalize probabilities */
    algorithmFPType invSumP = one / sumP;
    for (size_t j = 0; j < nClasses; j++)
    {
        p[j] *= invSumP;
    }
}


} // namespace internal
} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
