/* file: brownboost_train_impl.i */
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
//  Implementation of Freund method for Brown Boost training algorithm.
//--
*/
/*
//
//  REFERENCES
//
//  1. Y. Freund. An adaptive version of the boost by majority algorithm, 2000
//
*/

#ifndef __BROWNBOOST_TRAIN_IMPL_I__
#define __BROWNBOOST_TRAIN_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_math.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "weak_learner_model.h"
#include "brownboost_model.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace training
{
namespace internal
{

template <Method method, typename algorithmFPType, CpuType cpu>
void BrownBoostTrainKernel<method, algorithmFPType, cpu>::compute(size_t na, NumericTablePtr *a,
                                                                  Model *r, const Parameter *par)
{
    NumericTablePtr xTable = a[0];
    NumericTablePtr yTable = a[1];
    Parameter *parameter = const_cast<Parameter *>(par);
    r->setNFeatures(xTable->getNumberOfColumns());

    size_t nVectors  = xTable->getNumberOfRows();

    size_t nWeakLearners = 0;               /* Number of weak learners */
    algorithmFPType *alpha = NULL;          /* BrownBoost coefficients */

    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > hTable(
        new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(1, nVectors));
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > wTable(
        new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(1, nVectors));

    /* Get classification labels */
    algorithmFPType *y;             /* Classification labels */
    daal::internal::FeatureMicroTable<algorithmFPType, readOnly, cpu> mtY(yTable.get());
    mtY.getBlockOfColumnValues(0, 0, nVectors, &y);

    NumericTablePtr weakLearnerInputTables[] = {xTable, yTable, wTable};

    /* Run BrownBoost training */
    brownBoostFreundKernel(nVectors, weakLearnerInputTables, hTable, y, r, parameter, &nWeakLearners, &alpha);

    /* Update Brown Boost model with calculated results */
    NumericTablePtr alphaTable = r->getAlpha();

    algorithmFPType *resAlpha;

    alphaTable->setNumberOfRows(nWeakLearners);
    alphaTable->allocateDataMemory();
    daal::internal::FeatureMicroTable<algorithmFPType, writeOnly, cpu> mtAlpha(alphaTable.get());
    mtAlpha.getBlockOfColumnValues(0, 0, nWeakLearners, &resAlpha);

    for (size_t i = 0; i < nWeakLearners; i++)
    {
        resAlpha[i]  = alpha[i];
    }

    mtAlpha.release();
    if(!this->_errors->isEmpty()) { return; }

    mtY.release();
    if(!this->_errors->isEmpty()) { return; }

    if (alpha) { daal::services::daal_free(alpha); }
}

/**
 *  \brief BrownBoost algoritrhm kernel
 *
 *  \param nVectors[in]               Number of observations
 *  \param weakLearnerInputTables[in] Array of 3 numeric tables [xTable, yTable, wTable] needed to train weak learner.
 *                                    xTable - holds matrix of observations
 *                                    yTable - holds array of class labels, y[i] is from {-1, 1}
 *                                    wTable - holds array of obserwations' weights
 *  \param hTable[in,out]             Table to store weak learner's classificatiion results
 *  \param y[in]                      Array of classification labels
 *  \param h[in,out]                  Array of weak learner's classificatiion results
 *  \param w[in,out]                  Array of observations' weights
 *  \param boostModel[in]             Brown Boost model
 *  \param parameter[in]              Brown Boost parameters
 *  \param nWeakLearnersPtr[out]      Number of weak learners
 *  \param modelsPtr[out]             Resulting array of weak learners' models
 *  \param alphaPtr[out]              Resulting array of boosting coefficients
 *
  */
template <Method method, typename algorithmFPType, CpuType cpu>
void BrownBoostTrainKernel<method, algorithmFPType, cpu>::brownBoostFreundKernel(
    size_t nVectors, NumericTablePtr weakLearnerInputTables[],
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > hTable, algorithmFPType *y,
    Model *boostModel, Parameter *parameter, size_t *nWeakLearnersPtr,
    algorithmFPType **alphaPtr)
{
    algorithmFPType *alpha = NULL;          /* BrownBoost coefficients */
    algorithmFPType *r;                     /* Weak classifier's classification margin */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > wTable =
        services::staticPointerCast<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>, NumericTable>
        (weakLearnerInputTables[2]);
    algorithmFPType *w = wTable->getArray();
    algorithmFPType *h = hTable->getArray();

    /* Floating point constants */
    const algorithmFPType zero = (algorithmFPType)0.0;
    const algorithmFPType one  = (algorithmFPType)1.0;

    NewtonRaphsonKernel<method, algorithmFPType, cpu> nr(nVectors, parameter, _errors);
    if (!this->_errors->isEmpty()) { return; }

    /* Allocate memory for storing intermediate results */
    r = daal::services::internal::service_calloc<algorithmFPType, cpu>(nVectors);
    if (!r) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    services::SharedPtr<weak_learner::training::Batch>   learnerTrain   = parameter->weakLearnerTraining;
    learnerTrain->getErrors()->setCanThrow(false);
    learnerTrain->input.set(classifier::training::data,    weakLearnerInputTables[0]);
    learnerTrain->input.set(classifier::training::labels,  weakLearnerInputTables[1]);
    learnerTrain->input.set(classifier::training::weights, weakLearnerInputTables[2]);

    services::SharedPtr<weak_learner::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction;
    learnerPredict->inputBase->set(classifier::prediction::data, weakLearnerInputTables[0]);

    services::SharedPtr<classifier::prediction::Result> predictionRes(new classifier::prediction::Result());
    predictionRes->set(classifier::prediction::prediction, hTable);
    learnerPredict->setResult(predictionRes);

    /* Clear the collection of weak learners models in the boosting model */
    boostModel->clearWeakLearnerModels();

    algorithmFPType s = nr.c;      /* Remaining time */
    size_t nWeakLearners = 0;
    for (size_t iteration = 0; iteration < parameter->maxIterations && s > zero; iteration++)
    {
        nWeakLearners++;

        /* Update weights */
        updateWeights(nVectors, s, nr.c, nr.invSqrtC, r, nr.nra, nr.nre2, w);

        /* Re-allocate array of weak learners' models and boosting coefficients */
        alpha = reallocateAlpha(nWeakLearners-1, nWeakLearners, alpha);
        if (!this->_errors->isEmpty()) { daal::services::daal_free(r); return; }

        /* Make weak learner to allocate new memory for storing training result */
        if (iteration > 0) { learnerTrain->resetResult(); }

        /* Train weak learner's model */
        learnerTrain->computeNoThrow();
        if(learnerTrain->getErrors()->size() != 0) {daal::services::daal_free(r); this->_errors->add(learnerTrain->getErrors()->getErrors()); return;}

        services::SharedPtr<classifier::training::Result> trainingRes = learnerTrain->getResult();
        services::SharedPtr<weak_learner::Model> learnerModel =
            services::staticPointerCast<weak_learner::Model, classifier::Model>(trainingRes->get(classifier::training::model));
        boostModel->addWeakLearnerModel(learnerModel);

        /* Get weak learner's classification results */
        learnerPredict->inputBase->set(classifier::prediction::model, learnerModel);
        learnerPredict->computeNoThrow();
        if(learnerPredict->getErrors()->size() != 0) {daal::services::daal_free(r); this->_errors->add(learnerPredict->getErrors()->getErrors()); return;}

        algorithmFPType gamma = zero;

        size_t nCorrect = 0;
        for (size_t j = 0; j < nVectors; j++)
        {
            h[j] = ((h[j] > zero) ? one : -one);
            algorithmFPType hy = h[j] * y[j];
            gamma += w[j] * hy;
            nCorrect += (size_t)(hy > zero);
        }

        if (nCorrect == nVectors)
        {
            /* Here if and only if the first weak learner recognizes all objects correctly */
            /* Choose alpha[0] to make the predictions of BrownBoost classifier equal to +/-erf(4)
               which is close to +/-1 */
            alpha[nWeakLearners - 1] = 4.0 * nr.sqrtC;
            break;
        }

        /* Find alpha coefficient with Newton-Raphson method */
        nr.compute(gamma, s, h, y);
        s -= nr.nrT;
        alpha[nWeakLearners - 1] = nr.nrAlpha;

        /* Update margin */
        for (size_t j = 0; j < nVectors; j++)
        {
            r[j] += nr.nrAlpha * nr.nrb[j];
        }
    }

    *nWeakLearnersPtr = nWeakLearners;
    *alphaPtr  = alpha;

    daal::services::daal_free(r);
}

template <Method method, typename algorithmFPType, CpuType cpu>
void BrownBoostTrainKernel<method, algorithmFPType, cpu>::updateWeights(
            size_t nVectors, algorithmFPType s, algorithmFPType c, algorithmFPType invSqrtC,
            const algorithmFPType *r, algorithmFPType *nra, algorithmFPType *nre2, algorithmFPType *w)
{
    for (size_t j = 0; j < nVectors; j++)
    {
        nra[j] = r[j] + s;
        nre2[j] = nra[j] * invSqrtC;
        w[j] = -nra[j] * nra[j] / c;
    }
    daal::internal::Math<algorithmFPType,cpu>::vExp(nVectors, w, w);
    daal::internal::Math<algorithmFPType,cpu>::vErf(nVectors, nre2, nre2);
    algorithmFPType wSum = (algorithmFPType)0.0;
    for (size_t j = 0; j < nVectors; j++)
    {
        wSum += w[j];
    }
    algorithmFPType invWSum = 1.0 / wSum;
    for (size_t j = 0; j < nVectors; j++)
    {
        w[j] *= invWSum;
    }
}

template <Method method, typename algorithmFPType, CpuType cpu>
algorithmFPType* BrownBoostTrainKernel<method, algorithmFPType, cpu>::reallocateAlpha(
            size_t oldAlphaSize, size_t alphaSize, algorithmFPType *oldAlpha)
{
    algorithmFPType *alpha = (algorithmFPType *)daal::services::daal_malloc(alphaSize * sizeof(algorithmFPType));
    if (!alpha) { this->_errors->add(services::ErrorMemoryAllocationFailed); return NULL; }

    daal::services::daal_memcpy_s(alpha, alphaSize * sizeof(algorithmFPType), oldAlpha, oldAlphaSize * sizeof(algorithmFPType));

    if (oldAlpha) { daal::services::daal_free(oldAlpha); }
    return alpha;
}

template <Method method, typename algorithmFPType, CpuType cpu>
NewtonRaphsonKernel<method, algorithmFPType, cpu>::NewtonRaphsonKernel(
            size_t nVectors, Parameter *parameter, services::SharedPtr<services::KernelErrorCollection> _errors)
{
    this->nVectors = nVectors;
    this->_errors  = _errors;
    nra  = (algorithmFPType *)daal::services::daal_malloc(nVectors * sizeof(algorithmFPType));
    nrb  = (algorithmFPType *)daal::services::daal_malloc(nVectors * sizeof(algorithmFPType));
    nrd  = (algorithmFPType *)daal::services::daal_malloc(nVectors * sizeof(algorithmFPType));
    nrw  = (algorithmFPType *)daal::services::daal_malloc(nVectors * sizeof(algorithmFPType));
    nre1 = (algorithmFPType *)daal::services::daal_malloc(nVectors * sizeof(algorithmFPType));
    nre2 = (algorithmFPType *)daal::services::daal_malloc(nVectors * sizeof(algorithmFPType));
    if (!nra || !nrb || !nrd || !nrw || !nre1 || !nre2) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    const algorithmFPType one = (algorithmFPType)1.0;
    const algorithmFPType pi  = (algorithmFPType)3.1415926535897932384626433832795;

    error      = parameter->accuracyThreshold;
    nrAccuracy = parameter->newtonRaphsonAccuracyThreshold;
    nrMaxIter  = parameter->newtonRaphsonMaxIterations;
    nu         = parameter->degenerateCasesThreshold;
    sqrtC = daal::internal::Math<algorithmFPType,cpu>::sErfInv(one - error);
    c     = sqrtC * sqrtC;
    invC  = one / c;
    invSqrtC = one / sqrtC;
    sqrtPiC = daal::internal::Math<algorithmFPType,cpu>::sSqrt(pi * c);
}

template <Method method, typename algorithmFPType, CpuType cpu>
NewtonRaphsonKernel<method, algorithmFPType, cpu>::~NewtonRaphsonKernel()
{
    if (nra) daal::services::daal_free(nra);
    if (nrb) daal::services::daal_free(nrb);
    if (nrd) daal::services::daal_free(nrd);
    if (nrw) daal::services::daal_free(nrw);
    if (nre1) daal::services::daal_free(nre1);
    if (nre2) daal::services::daal_free(nre2);
}

template <Method method, typename algorithmFPType, CpuType cpu>
void NewtonRaphsonKernel<method, algorithmFPType, cpu>::compute(algorithmFPType gamma, algorithmFPType s,
            algorithmFPType *h, algorithmFPType *y)
{
    /* Floating point constants */
    const algorithmFPType zero = (algorithmFPType)0.0;
    const algorithmFPType one  = (algorithmFPType)1.0;
    const algorithmFPType two  = (algorithmFPType)2.0;

    algorithmFPType alphaSign = one;
    if (gamma < zero)
    {
        /* Here if weak classifier is worse that random guessing.
           If we invert predictions, weak classifier became better then random guessing */
        gamma = zero - gamma;
        alphaSign = -one;
    }

    /* Find alpha coefficients with Newton-Raphson method */
    algorithmFPType nrW, nrU, nrB, nrV, nrE;

    nrAlpha = ((error < gamma) ? error : gamma);
    nrT     = nrAlpha * nrAlpha / 3.0;

    for (size_t j = 0; j < nVectors; j++)
    {
        nrb[j] = h[j] * y[j] * alphaSign;
    }

    bool nrDone = false;
    for (size_t nrIter = 0; !nrDone && nrIter < nrMaxIter; nrIter++)
    {
        /* Calculate Newton-Raphson parameters */
        nrW = zero;
        nrU = zero;
        nrB = zero;
        nrV = zero;
        nrE = zero;
        for (size_t j = 0; j < nVectors; j++)
        {
            nrd[j] = nra[j] + nrAlpha * nrb[j] - nrT;
            nrw[j] = -invC * nrd[j] * nrd[j];
            nre1[j] = nrd[j] * invSqrtC;
        }
        daal::internal::Math<algorithmFPType,cpu>::vExp(nVectors, nrw,  nrw);
        daal::internal::Math<algorithmFPType,cpu>::vErf(nVectors, nre1, nre1);
        for (size_t j = 0; j < nVectors; j++)
        {
            algorithmFPType nrwb  = nrw[j] * nrb[j];
            algorithmFPType nrwdb = nrwb * nrd[j];
            nrW += nrw[j];
            nrB += nrwb;
            nrU += nrwdb;
            nrV += nrwdb * nrb[j];
            nrE += nre1[j] - nre2[j];
        }

        /* Update Newton-Raphson variables */
        algorithmFPType invDenom = one / (two * (nrV * nrW - nrU * nrB));
        nrAlpha += invDenom * (c * nrW * nrB + sqrtPiC * nrU * nrE);
        nrT     += invDenom * (c * nrB * nrB + sqrtPiC * nrV * nrE);

        if (daal::internal::Math<algorithmFPType,cpu>::sFabs(nrB / nrW) <= nu) { nrDone = true; }
        if (daal::internal::Math<algorithmFPType,cpu>::sFabs(nrB) <= nrAccuracy &&
            daal::internal::Math<algorithmFPType,cpu>::sFabs(nrE) <= nrAccuracy) { nrDone = true; }
    }

    nrAlpha *= alphaSign;
}


} // namespace daal::algorithms::brownboost::training::internal
}
}
}
} // namespace daal

#endif
