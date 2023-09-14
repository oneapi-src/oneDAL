/* file: brownboost_train_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"

#include "algorithms/weak_learner/weak_learner_model.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/boosting/brownboost_model.h"
#include "src/algorithms/brownboost/brownboost_train_kernel.h"

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
using namespace daal::internal;
using namespace daal::data_management;

template <Method method, typename algorithmFPType, CpuType cpu>

services::Status BrownBoostTrainKernel<method, algorithmFPType, cpu>::compute(size_t na, NumericTablePtr * a, Model * r, const Parameter * par)
{
    NumericTablePtr xTable = a[0];
    NumericTablePtr yTable = a[1];
    Parameter * parameter  = const_cast<Parameter *>(par);
    r->setNFeatures(xTable->getNumberOfColumns());

    const size_t nVectors = xTable->getNumberOfRows();

    size_t nWeakLearners    = 0;       /* Number of weak learners */
    algorithmFPType * alpha = nullptr; /* BrownBoost coefficients */

    services::Status s;
    HomogenNTPtr hTable = HomogenNT::create(1, nVectors, &s);
    DAAL_CHECK_STATUS_VAR(s);
    HomogenNTPtr wTable = HomogenNT::create(1, nVectors, &s);
    DAAL_CHECK_STATUS_VAR(s);

    NumericTablePtr weakLearnerInputTables[] = { xTable, yTable, wTable };

    /* Run BrownBoost training */
    {
        /* Get classification labels */
        ReadColumns<algorithmFPType, cpu> mtY(*yTable, 0, 0, nVectors);
        DAAL_CHECK_BLOCK_STATUS(mtY);
        DAAL_ASSERT(mtY.get());
        DAAL_CHECK_STATUS(s, brownBoostFreundKernel(nVectors, weakLearnerInputTables, hTable, mtY.get(), r, parameter, nWeakLearners, alpha));
    }

    /* Update Brown Boost model with calculated results */
    NumericTablePtr alphaTable = r->getAlpha();
    s                          = alphaTable->resize(nWeakLearners);
    if (s)
    {
        WriteOnlyColumns<algorithmFPType, cpu> mtAlpha(*alphaTable, 0, 0, nWeakLearners);
        s = mtAlpha.status();
        if (s)
        {
            algorithmFPType * resAlpha = mtAlpha.get();
            DAAL_ASSERT(resAlpha);
            for (size_t i = 0; i < nWeakLearners; i++) resAlpha[i] = alpha[i];
        }
    }

    if (alpha)
    {
        daal::services::daal_free(alpha);
        alpha = nullptr;
    }
    return s;
}

/**
 *  \brief BrownBoost algorithm kernel
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
 *  \param nWeakLearners[out]         Number of weak learners
 *  \param alpha[out]                 Resulting array of boosting coefficients
 *
  */
template <Method method, typename algorithmFPType, CpuType cpu>
services::Status BrownBoostTrainKernel<method, algorithmFPType, cpu>::brownBoostFreundKernel(size_t nVectors,
                                                                                             NumericTablePtr weakLearnerInputTables[],
                                                                                             const HomogenNTPtr & hTable, const algorithmFPType * y,
                                                                                             Model * boostModel, Parameter * parameter,
                                                                                             size_t & nWeakLearners, algorithmFPType *& alpha)
{
    alpha               = nullptr; /* BrownBoost coefficients */
    algorithmFPType * w = static_cast<HomogenNT *>(weakLearnerInputTables[2].get())->getArray();
    algorithmFPType * h = hTable->getArray();

    /* Floating point constants */
    const algorithmFPType zero = (algorithmFPType)0.0;
    const algorithmFPType one  = (algorithmFPType)1.0;

    NewtonRaphsonKernel<method, algorithmFPType, cpu> nr(nVectors, parameter->accuracyThreshold, parameter->newtonRaphsonAccuracyThreshold,
                                                         parameter->newtonRaphsonMaxIterations, parameter->degenerateCasesThreshold);
    DAAL_CHECK(nr.isValid(), services::ErrorMemoryAllocationFailed);

    /* Allocate memory for storing intermediate results */
    daal::internal::TArray<algorithmFPType, cpu> r(nVectors); /* Weak classifier's classification margin */
    DAAL_CHECK(r.get(), services::ErrorMemoryAllocationFailed);
    for (auto i = 0; i < nVectors; r[i++] = 0.)
        ;

    services::SharedPtr<classifier::training::Batch> learnerTrain = parameter->weakLearnerTraining->clone();
    classifier::training::Input * trainInput                      = learnerTrain->getInput();
    DAAL_CHECK(trainInput, services::ErrorNullInput);
    trainInput->set(classifier::training::data, weakLearnerInputTables[0]);
    trainInput->set(classifier::training::labels, weakLearnerInputTables[1]);
    trainInput->set(classifier::training::weights, weakLearnerInputTables[2]);

    services::SharedPtr<classifier::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction->clone();
    classifier::prediction::Input * predictInput                      = learnerPredict->getInput();
    DAAL_CHECK(predictInput, services::ErrorNullInput);
    predictInput->set(classifier::prediction::data, weakLearnerInputTables[0]);

    classifier::prediction::ResultPtr predictionRes(new classifier::prediction::Result());
    DAAL_CHECK_MALLOC(predictionRes.get())
    predictionRes->set(classifier::prediction::prediction, hTable);
    learnerPredict->setResult(predictionRes);

    /* Clear the collection of weak learners models in the boosting model */
    boostModel->clearWeakLearnerModels();

    algorithmFPType s = nr.c; /* Remaining time */
    nWeakLearners     = 0;
    services::Status status;
    for (size_t iteration = 0; iteration < parameter->maxIterations && s > zero; iteration++)
    {
        nWeakLearners++;

        /* Update weights */
        updateWeights(nVectors, s, nr.c, nr.invSqrtC, r.get(), nr.aNra.get(), nr.aNre2.get(), w);

        /* Re-allocate array of weak learners' models and boosting coefficients */
        alpha = reallocateAlpha(nWeakLearners - 1, nWeakLearners, alpha, status);
        if (!alpha) return services::Status(services::ErrorMemoryAllocationFailed);

        /* Make weak learner to allocate new memory for storing training result */
        if (iteration > 0)
        {
            learnerTrain->resetResult();
        }

        /* Train weak learner's model */
        DAAL_CHECK_STATUS(status, learnerTrain->computeNoThrow());

        classifier::training::ResultPtr trainingRes = learnerTrain->getResult();
        classifier::ModelPtr learnerModel =
            services::staticPointerCast<classifier::Model, classifier::Model>(trainingRes->get(classifier::training::model));
        boostModel->addWeakLearnerModel(learnerModel);

        /* Get weak learner's classification results */
        predictInput->set(classifier::prediction::model, learnerModel);
        DAAL_CHECK_STATUS(status, learnerPredict->computeNoThrow());

        algorithmFPType gamma = zero;

        size_t nCorrect = 0;
        for (size_t j = 0; j < nVectors; j++)
        {
            h[j]               = ((h[j] > zero) ? one : -one);
            algorithmFPType hy = h[j] * y[j];
            gamma += w[j] * hy;
            nCorrect += (hy > zero) ? 1 : 0;
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
        const algorithmFPType * nrb = nr.aNrb.get();
        algorithmFPType * rr        = r.get();
        for (size_t j = 0; j < nVectors; j++)
        {
            rr[j] += nr.nrAlpha * nrb[j];
        }
    }

    return status;
}

template <Method method, typename algorithmFPType, CpuType cpu>
void BrownBoostTrainKernel<method, algorithmFPType, cpu>::updateWeights(size_t nVectors, algorithmFPType s, algorithmFPType c,
                                                                        algorithmFPType invSqrtC, const algorithmFPType * r, algorithmFPType * nra,
                                                                        algorithmFPType * nre2, algorithmFPType * w)
{
    for (size_t j = 0; j < nVectors; j++)
    {
        nra[j]  = r[j] + s;
        nre2[j] = nra[j] * invSqrtC;
        w[j]    = -nra[j] * nra[j] / c;
    }
    daal::internal::MathInst<algorithmFPType, cpu>::vExp(nVectors, w, w);
    daal::internal::MathInst<algorithmFPType, cpu>::vErf(nVectors, nre2, nre2);
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
algorithmFPType * BrownBoostTrainKernel<method, algorithmFPType, cpu>::reallocateAlpha(size_t oldAlphaSize, size_t alphaSize,
                                                                                       algorithmFPType * oldAlpha, services::Status & s)
{
    algorithmFPType * alpha = (algorithmFPType *)daal::services::daal_malloc(alphaSize * sizeof(algorithmFPType));
    if (alpha && oldAlpha)
    {
        int result = 0;
        result =
            daal::services::internal::daal_memcpy_s(alpha, alphaSize * sizeof(algorithmFPType), oldAlpha, oldAlphaSize * sizeof(algorithmFPType));
        if (result)
        {
            s |= services::Status(services::ErrorMemoryCopyFailedInternal);
        }
    }
    if (oldAlpha)
    {
        daal::services::daal_free(oldAlpha);
        oldAlpha = nullptr;
    }
    return alpha;
}

template <Method method, typename algorithmFPType, CpuType cpu>
NewtonRaphsonKernel<method, algorithmFPType, cpu>::NewtonRaphsonKernel(size_t nVect, double parAccuracyThreshold,
                                                                       double parNewtonRaphsonAccuracyThreshold, double parNewtonRaphsonMaxIterations,
                                                                       double parDegenerateCasesThreshold)
    : nVectors(nVect),
      aNra(nVectors),
      aNrb(nVectors),
      aNrd(nVectors),
      aNrw(nVectors),
      aNre1(nVectors),
      aNre2(nVectors),
      error(parAccuracyThreshold),
      nrAccuracy(parNewtonRaphsonAccuracyThreshold),
      nrMaxIter(parNewtonRaphsonMaxIterations),
      nu(parDegenerateCasesThreshold)
{
    const algorithmFPType one = (algorithmFPType)1.0;
    const algorithmFPType pi  = (algorithmFPType)3.1415926535897932384626433832795;
    sqrtC                     = daal::internal::MathInst<algorithmFPType, cpu>::sErfInv(one - error);
    c                         = sqrtC * sqrtC;
    invC                      = one / c;
    invSqrtC                  = one / sqrtC;
    sqrtPiC                   = daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(pi * c);
}

template <Method method, typename algorithmFPType, CpuType cpu>
void NewtonRaphsonKernel<method, algorithmFPType, cpu>::compute(algorithmFPType gamma, algorithmFPType s, const algorithmFPType * h,
                                                                const algorithmFPType * y)
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
        gamma     = zero - gamma;
        alphaSign = -one;
    }

    /* Find alpha coefficients with Newton-Raphson method */

    nrAlpha = ((error < gamma) ? error : gamma);
    nrT     = nrAlpha * nrAlpha / 3.0;

    algorithmFPType * nrd  = aNrd.get();
    algorithmFPType * nrw  = aNrw.get();
    algorithmFPType * nra  = aNra.get();
    algorithmFPType * nrb  = aNrb.get();
    algorithmFPType * nre1 = aNre1.get();
    algorithmFPType * nre2 = aNre2.get();

    for (size_t j = 0; j < nVectors; j++)
    {
        nrb[j] = h[j] * y[j] * alphaSign;
    }

    for (size_t nrIter = 0; nrIter < nrMaxIter; ++nrIter)
    {
        /* Calculate Newton-Raphson parameters */
        for (size_t j = 0; j < nVectors; j++)
        {
            nrd[j]  = nra[j] + nrAlpha * nrb[j] - nrT;
            nrw[j]  = -invC * nrd[j] * nrd[j];
            nre1[j] = nrd[j] * invSqrtC;
        }
        daal::internal::MathInst<algorithmFPType, cpu>::vExp(nVectors, nrw, nrw);
        daal::internal::MathInst<algorithmFPType, cpu>::vErf(nVectors, nre1, nre1);
        algorithmFPType nrW(0.0);
        algorithmFPType nrU(0.0);
        algorithmFPType nrB(0.0);
        algorithmFPType nrV(0.0);
        algorithmFPType nrE(0.0);
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
        const algorithmFPType invDenom = one / (two * (nrV * nrW - nrU * nrB));
        nrAlpha += invDenom * (c * nrW * nrB + sqrtPiC * nrU * nrE);
        nrT += invDenom * (c * nrB * nrB + sqrtPiC * nrV * nrE);

        if ((daal::internal::MathInst<algorithmFPType, cpu>::sFabs(nrB / nrW) <= nu)
            || (daal::internal::MathInst<algorithmFPType, cpu>::sFabs(nrB) <= nrAccuracy
                && daal::internal::MathInst<algorithmFPType, cpu>::sFabs(nrE) <= nrAccuracy))
            break;
    }
    nrAlpha *= alphaSign;
}

} // namespace internal
} // namespace training
} // namespace brownboost
} // namespace algorithms
} // namespace daal

#endif
