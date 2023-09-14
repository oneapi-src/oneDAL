/* file: adaboost_train_impl.i */
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
//  Implementation of Freund method for Ada Boost training algorithm.
//--
*/
/*
//
//  REFERENCES
//
//  1. Robert E. Schapire. Explaining AdaBoost, Springer, 2013
//     http://rob.schapire.net/papers/explaining-adaboost.pdf
//  2. K. V. Vorontsov. Lectures about algorithm ensembles.
//     http://www.machinelearning.ru/wiki/images/0/0d/Voron-ML-Compositions.pdf
//
*/

#ifndef __ADABOOST_TRAIN_IMPL_I__
#define __ADABOOST_TRAIN_IMPL_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_data_utils.h"

#include "algorithms/classifier/classifier_model.h"
#include "algorithms/boosting/adaboost_model.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::algorithms;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{
namespace internal
{
using namespace daal::internal;

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostTrainKernel<method, algorithmFPType, cpu>::adaboostSAMME(size_t nVectors, NumericTablePtr weakLearnerInputTables[],
                                                                                  const algorithmFPType * y, Model * boostModel,
                                                                                  algorithmFPType * weakLearnersErrorsArray,
                                                                                  const Parameter * parameter, size_t & nWeakLearners,
                                                                                  algorithmFPType * alpha)
{
    algorithmFPType * w = static_cast<HomogenNT *>(weakLearnerInputTables[2].get())->getArray();

    services::Status s;
    HomogenNTPtr hTable(HomogenNT::create(1, nVectors, &s));
    DAAL_CHECK_STATUS_VAR(s);
    algorithmFPType * h = hTable->getArray();

    /* Floating point constants */
    const algorithmFPType zero        = (algorithmFPType)0.0;
    const algorithmFPType one         = (algorithmFPType)1.0;
    const algorithmFPType invNVectors = one / (algorithmFPType)nVectors;

    /* Get number of AdaBoost iterations */
    const size_t maxIter               = parameter->maxIterations;
    const size_t accThr                = parameter->accuracyThreshold;
    const algorithmFPType learningRate = parameter->learningRate;

    /* Allocate memory for storing intermediate results */
    /* Vector of flags.
    errFlag[i] == -1.0, if weak classifier's classification result agrees with actual class label;
    errFlag[i] == 1.0, otherwise */
    TArray<algorithmFPType, cpu> aErrFlag(nVectors);
    DAAL_CHECK(aErrFlag.get(), services::ErrorMemoryAllocationFailed);

    /* Initialize weights */
    service_memset<algorithmFPType, cpu>(w, invNVectors, nVectors);

    services::SharedPtr<classifier::training::Batch> learnerTrain = parameter->weakLearnerTraining->clone();
    learnerTrain->enableChecks(false);
    const size_t nClasses                    = parameter->nClasses;
    classifier::training::Input * trainInput = learnerTrain->getInput();
    DAAL_CHECK(trainInput, services::ErrorNullInput);
    trainInput->set(classifier::training::data, weakLearnerInputTables[0]);
    trainInput->set(classifier::training::labels, weakLearnerInputTables[1]);
    trainInput->set(classifier::training::weights, weakLearnerInputTables[2]);

    services::SharedPtr<classifier::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction->clone();
    learnerPredict->enableChecks(false);
    classifier::prediction::Input * predictInput = learnerPredict->getInput();
    DAAL_CHECK(predictInput, services::ErrorNullInput);
    predictInput->set(classifier::prediction::data, weakLearnerInputTables[0]);

    classifier::prediction::ResultPtr predictionRes(new classifier::prediction::Result());
    DAAL_CHECK_MALLOC(predictionRes.get())
    predictionRes->set(classifier::prediction::prediction, NumericTablePtr(hTable));
    learnerPredict->setResult(predictionRes);

    nWeakLearners            = 0;
    algorithmFPType maxAlpha = zero;

    /* Clear the collection of weak learners models in the boosting model */
    boostModel->clearWeakLearnerModels();

    for (size_t m = 0; m < maxIter; m++)
    {
        nWeakLearners++;

        /* Make weak learner to allocate new memory for storing training result */
        if (m > 0)
        {
            learnerTrain->resetResult();
        }

        /* Train weak learner's model */
        DAAL_CHECK_STATUS(s, learnerTrain->computeNoThrow());

        classifier::training::ResultPtr trainingRes = learnerTrain->getResult();
        classifier::ModelPtr learnerModel           = trainingRes->get(classifier::training::model);

        /* Add new model to the collection of the boosting algorithm models */
        boostModel->addWeakLearnerModel(learnerModel);

        /* Get weak learner's classification results */
        predictInput->set(classifier::prediction::model, learnerModel);
        DAAL_CHECK_STATUS(s, learnerPredict->computeNoThrow());

        /* Calculate weighted error and errFlag: product of predicted * ground_truth */
        size_t nErr               = 0;
        algorithmFPType errM      = zero;
        algorithmFPType * errFlag = aErrFlag.get();
        for (size_t i = 0; i < nVectors; i++)
        {
            errFlag[i] = zero;
            if (h[i] != y[i])
            {
                errFlag[i] = one;
                nErr++;
                errM += w[i];
            }
        }
        if (weakLearnersErrorsArray)
        {
            weakLearnersErrorsArray[m] = errM;
        }

        if (nErr == 0)
        {
            /* Here if weak learner's classification error is 0 */
            alpha[m] = ((maxAlpha > zero) ? maxAlpha : one);
            break;
        }
        if (nErr == nVectors)
        {
            /* Here if weak learner's classification error is 1 */
            alpha[m] = 0.0;
            nWeakLearners--;
            break;
        }

        algorithmFPType cM =
            learningRate * (MathInst<algorithmFPType, cpu>::sLog((one - errM) / errM) + MathInst<algorithmFPType, cpu>::sLog(nClasses - one));

        /* Update weights */
        for (size_t i = 0; i < nVectors; i++)
        {
            errFlag[i] *= cM;
        }
        MathInst<algorithmFPType, cpu>::vExp(nVectors, errFlag, errFlag);
        algorithmFPType wSum = zero;
        for (size_t i = 0; i < nVectors; i++)
        {
            w[i] *= errFlag[i];
            wSum += w[i];
        }
        algorithmFPType invWSum = one / wSum;
        for (size_t i = 0; i < nVectors; i++)
        {
            w[i] *= invWSum;
        }
        alpha[m] = cM;
        if (errM < accThr)
        {
            break;
        }
        if (alpha[m] > maxAlpha)
        {
            maxAlpha = alpha[m];
        }
    }
    return s;
}

template <Method method, typename algorithmFPType, CpuType cpu>
void AdaBoostTrainKernel<method, algorithmFPType, cpu>::convertLabelToVector(size_t nClasses, algorithmFPType * Y)
{
    const algorithmFPType nonClassValue = -1.0 / (nClasses - 1.0);
    for (size_t j = 0; j < (nClasses + 1) * nClasses; j++)
    {
        Y[j] = nonClassValue;
    }

    const algorithmFPType one   = 1.0;
    Y[0]                        = one;
    algorithmFPType * Y_shifted = &Y[nClasses];
    for (size_t i = 0; i < nClasses; i++)
    {
        Y_shifted[i * nClasses + i] = one;
    }
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostTrainKernel<method, algorithmFPType, cpu>::adaboostSAMME_R(size_t nVectors, NumericTablePtr weakLearnerInputTables[],
                                                                                    const algorithmFPType * y, Model * boostModel,
                                                                                    algorithmFPType * weakLearnersErrorsArray,
                                                                                    const Parameter * parameter, size_t & nWeakLearners,
                                                                                    algorithmFPType * alpha)
{
    services::Status s;
    algorithmFPType * w                = static_cast<HomogenNT *>(weakLearnerInputTables[2].get())->getArray();
    const size_t nClasses              = parameter->nClasses;
    const algorithmFPType learningRate = parameter->learningRate;

    HomogenNTPtr pTable(HomogenNT::create(nClasses, nVectors, &s));
    DAAL_CHECK_STATUS_VAR(s);
    algorithmFPType * p = pTable->getArray();

    HomogenNTPtr tTable(HomogenNT::create(1, nVectors, &s));
    DAAL_CHECK_STATUS_VAR(s);
    algorithmFPType * t = tTable->getArray();

    HomogenNTPtr yTable(HomogenNT::create(nClasses, nClasses + 1, &s));
    DAAL_CHECK_STATUS_VAR(s);
    algorithmFPType * Y_start = yTable->getArray();
    convertLabelToVector(nClasses, Y_start);
    algorithmFPType * Y = &Y_start[nClasses]; // shifted array to handle -1 label, Y[-1,:] == Y[0,:]

    /* Floating point constants */
    const algorithmFPType zero        = (algorithmFPType)0.0;
    const algorithmFPType one         = (algorithmFPType)1.0;
    const algorithmFPType invNVectors = one / (algorithmFPType)nVectors;

    /* Initialize weights */
    for (size_t i = 0; i < nVectors; i++)
    {
        w[i] = invNVectors;
    }

    services::SharedPtr<classifier::training::Batch> learnerTrain = parameter->weakLearnerTraining->clone();
    learnerTrain->enableChecks(false);
    classifier::training::Input * trainInput = learnerTrain->getInput();
    DAAL_CHECK(trainInput, services::ErrorNullInput);
    trainInput->set(classifier::training::data, weakLearnerInputTables[0]);
    trainInput->set(classifier::training::labels, weakLearnerInputTables[1]);
    trainInput->set(classifier::training::weights, weakLearnerInputTables[2]);

    services::SharedPtr<classifier::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction->clone();
    learnerPredict->enableChecks(false);
    classifier::prediction::Input * predictInput = learnerPredict->getInput();
    DAAL_CHECK(predictInput, services::ErrorNullInput);
    predictInput->set(classifier::prediction::data, weakLearnerInputTables[0]);

    classifier::prediction::ResultPtr predictionRes(new classifier::prediction::Result());
    DAAL_CHECK_MALLOC(predictionRes.get())
    predictionRes->set(classifier::prediction::probabilities, NumericTablePtr(pTable));
    learnerPredict->setResult(predictionRes);
    learnerPredict->parameter().resultsToEvaluate = classifier::computeClassProbabilities;

    nWeakLearners = 0;

    /* Clear the collection of weak learners models in the boosting model */
    boostModel->clearWeakLearnerModels();

    const algorithmFPType eps     = services::internal::EpsilonVal<algorithmFPType>::get();
    const algorithmFPType scaling = -(learningRate * (nClasses - one) / (nClasses));

    const size_t maxIter = parameter->maxIterations;
    for (size_t m = 0; m < maxIter; m++)
    {
        nWeakLearners++;

        /* Make weak learner to allocate new memory for storing training result */
        if (m > 0)
        {
            learnerTrain->resetResult();
        }

        /* Train weak learner's model */
        DAAL_CHECK_STATUS(s, learnerTrain->computeNoThrow());

        classifier::training::ResultPtr trainingRes = learnerTrain->getResult();
        classifier::ModelPtr learnerModel           = trainingRes->get(classifier::training::model);

        /* Add new model to the collection of the boosting algorithm models */
        boostModel->addWeakLearnerModel(learnerModel);

        /* Get weak learner's classification results */
        predictInput->set(classifier::prediction::model, learnerModel);
        DAAL_CHECK_STATUS(s, learnerPredict->computeNoThrow());

        algorithmFPType wSum = zero;
        for (size_t i = 0; i < nVectors * nClasses; i++)
        {
            if (p[i] < eps)
            {
                p[i] = eps;
            }
        }
        for (size_t i = 0; i < nVectors; i++)
        {
            t[i] = zero;
            for (size_t j = 0; j < nClasses; j++)
            {
                t[i] += Y[((int)y[i]) * nClasses + j] * p[i * nClasses + j];
            }
            t[i] *= scaling;
        }
        MathInst<algorithmFPType, cpu>::vExp(nVectors, t, t);
        for (size_t i = 0; i < nVectors; i++)
        {
            w[i] *= t[i];
            wSum += w[i];
        }

        algorithmFPType invWSum = one / wSum;
        for (size_t i = 0; i < nVectors; i++)
        {
            w[i] *= invWSum;
        }
        alpha[m] = 1.0;

        if (weakLearnersErrorsArray)
        {
            algorithmFPType errM = zero;
            for (size_t i = 0; i < nVectors; i++)
            {
                algorithmFPType pMax = zero;
                size_t iMax          = 0;
                for (size_t j = 0; j < nClasses; j++)
                {
                    if (p[i * nClasses + j] > pMax)
                    {
                        pMax = p[i * nClasses + j];
                        iMax = j;
                    }
                }
                if (nClasses > 2)
                {
                    if (iMax != y[i])
                    {
                        errM += w[i];
                    }
                }
                else
                {
                    if ((iMax == 0 && y[i] == 1) || (iMax == 1 && y[i] == -1))
                    {
                        errM += w[i];
                    }
                }
            }
            weakLearnersErrorsArray[m] = errM;
        }
    }
    return s;
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostTrainKernel<method, algorithmFPType, cpu>::compute(NumericTablePtr * a, Model * r, NumericTable * weakLearnersErrorsTable,
                                                                            const Parameter * parameter)
{
    NumericTablePtr xTable = a[0];
    NumericTablePtr yTable = a[1];
    r->setNFeatures(xTable->getNumberOfColumns());

    const size_t nVectors = xTable->getNumberOfRows();

    size_t nWeakLearners = 0; /* Number of weak learners */

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, parameter->maxIterations, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors, sizeof(algorithmFPType));

    /* Allocate memory for storing weak learners' models and boosting coefficients */
    TArray<algorithmFPType, cpu> alpha(parameter->maxIterations); /* AdaBoost coefficients */
    DAAL_CHECK(alpha.get(), services::ErrorMemoryAllocationFailed);

    services::Status s;
    HomogenNTPtr wTable(HomogenNT::create(1, nVectors, &s));
    DAAL_CHECK_STATUS_VAR(s);

    NumericTablePtr weakLearnerInputTables[] = { xTable, yTable, wTable };

    /* Run AdaBoost training */
    {
        /* Get classification labels */
        ReadColumns<algorithmFPType, cpu> mtY(*yTable, 0, 0, nVectors);
        DAAL_CHECK_BLOCK_STATUS(mtY);
        DAAL_ASSERT(mtY.get());

        WriteOnlyRows<algorithmFPType, cpu> mtWLErrors(weakLearnersErrorsTable);
        algorithmFPType * weakLearnersErrorsArray = mtWLErrors.next(0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtWLErrors);
        if (method == samme)
        {
            DAAL_CHECK_STATUS(
                s, adaboostSAMME(nVectors, weakLearnerInputTables, mtY.get(), r, weakLearnersErrorsArray, parameter, nWeakLearners, alpha.get()));
        }
        else
        {
            DAAL_CHECK_STATUS(
                s, adaboostSAMME_R(nVectors, weakLearnerInputTables, mtY.get(), r, weakLearnersErrorsArray, parameter, nWeakLearners, alpha.get()));
        }
    }

    /* Update Ada Boost model with calculated results */
    NumericTablePtr alphaTable = r->getAlpha();
    DAAL_CHECK_STATUS(s, alphaTable->resize(nWeakLearners));
    WriteOnlyColumns<algorithmFPType, cpu> mtAlpha(*alphaTable, 0, 0, nWeakLearners);
    DAAL_CHECK_BLOCK_STATUS(mtAlpha);
    algorithmFPType * resAlpha = mtAlpha.get();
    DAAL_ASSERT(resAlpha);
    for (size_t i = 0; i < nWeakLearners; i++)
    {
        resAlpha[i] = alpha[i];
    }
    return s;
}
} // namespace internal
} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif
