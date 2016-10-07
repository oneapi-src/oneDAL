/* file: adaboost_train_impl.i */
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

#include "algorithm.h"
#include "numeric_table.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_math.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "weak_learner_model.h"
#include "adaboost_model.h"

using namespace daal::data_management;

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

/**
 *  \brief AdaBoost algorithm kernel
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
 *  \param boostModel[in]             Ada Boost model
 *  \param parameter[in]              Ada Boost parameters
 *  \param nWeakLearnersPtr[out]      Number of weak learners
 *  \param modelsPtr[out]             Resulting array of weak learners' models
 *  \param alphaPtr[out]              Resulting array of boosting coefficients
 *
 */
template <Method method, typename algorithmFPType, CpuType cpu>
void AdaBoostTrainKernel<method, algorithmFPType, cpu>::adaBoostFreundKernel(
    size_t nVectors, NumericTablePtr weakLearnerInputTables[],
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > hTable, algorithmFPType *y,
    Model *boostModel, Parameter *parameter, size_t *nWeakLearnersPtr,
    algorithmFPType **alphaPtr)
{
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > wTable =
        services::staticPointerCast<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>, NumericTable>
        (weakLearnerInputTables[2]);
    algorithmFPType *w = wTable->getArray();
    algorithmFPType *h = hTable->getArray();

    algorithmFPType *errFlag;   /* Vector of flags.
                                   errFlag[i] == 0.0, if weak classifier's classification result agrees with actual class label;
                                   errFlag[i] == 1.0, otherwise */

    /* Floating point constants */
    const algorithmFPType zero = (algorithmFPType)0.0;
    const algorithmFPType one  = (algorithmFPType)1.0;
    const algorithmFPType invNVectors = one / (algorithmFPType)nVectors;

    /* Get number of AdaBoost iterations */
    const size_t maxIter = parameter->maxIterations;
    const size_t accThr  = parameter->accuracyThreshold;

    /* Allocate memory for storing weak learners' models and boosting coefficients */
    algorithmFPType *alpha = (algorithmFPType *) daal::services::daal_malloc(maxIter * sizeof(algorithmFPType));
    if (!alpha) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Allocate memory for storing intermediate results */
    errFlag = (algorithmFPType *) daal::services::daal_malloc(nVectors * sizeof(algorithmFPType), 64);
    if (!errFlag) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Initialize weights */
    for (size_t i = 0; i < nVectors; i++)
    {
        w[i] = invNVectors;
    }

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

    size_t nWeakLearners = 0;
    algorithmFPType maxAlpha = zero;

    /* Clear the collection of weak learners models in the boosting model */
    boostModel->clearWeakLearnerModels();

    for (size_t m = 0; m < maxIter; m++)
    {
        nWeakLearners++;

        /* Make weak learner to allocate new memory for storing training result */
        if (m > 0) { learnerTrain->resetResult(); }

        /* Train weak learner's model */
        learnerTrain->computeNoThrow();
        if(learnerTrain->getErrors()->size() != 0) {daal::services::daal_free(errFlag); this->_errors->add(learnerTrain->getErrors()->getErrors()); return;}

        services::SharedPtr<classifier::training::Result> trainingRes = learnerTrain->getResult();
        services::SharedPtr<weak_learner::Model> learnerModel =
            services::staticPointerCast<weak_learner::Model, classifier::Model>(trainingRes->get(classifier::training::model));

        /* Add new model to the collection of the boosting algorithm models */
        boostModel->addWeakLearnerModel(learnerModel);

        /* Get weak learner's classification results */
        learnerPredict->inputBase->set(classifier::prediction::model, learnerModel);
        learnerPredict->computeNoThrow();
        if(learnerPredict->getErrors()->size() != 0) {daal::services::daal_free(errFlag); this->_errors->add(learnerPredict->getErrors()->getErrors()); return;}

        /* Calculate weighted error and errFlag: product of predicted * ground_truth */
        size_t nErr = 0;
        algorithmFPType errM = zero;
        for (size_t i = 0; i < nVectors; i++)
        {
            errFlag[i] = -one;
            if (h[i] * y[i] < zero)
            {
                errFlag[i] = one;
                nErr++;
                errM += w[i];
            }
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

        algorithmFPType cM = 0.5 * daal::internal::Math<algorithmFPType,cpu>::sLog((one - errM) / errM);

        /* Update weights */
        for (size_t i = 0; i < nVectors; i++)
        {
            errFlag[i] *= cM;
        }
        daal::internal::Math<algorithmFPType,cpu>::vExp(nVectors, errFlag, errFlag);
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

        if (errM < accThr) { break; }
        if (alpha[m] > maxAlpha) { maxAlpha = alpha[m]; }
    }

    *nWeakLearnersPtr = nWeakLearners;
    *alphaPtr  = alpha;

    daal::services::daal_free(errFlag);
    return;
}

template <Method method, typename algorithmFPType, CpuType cpu>
void AdaBoostTrainKernel<method, algorithmFPType, cpu>::compute(size_t na, NumericTablePtr *a,
                                                                Model *r, const Parameter *par)
{
    NumericTablePtr xTable = a[0];
    NumericTablePtr yTable = a[1];
    Parameter *parameter = const_cast<Parameter *>(par);
    r->setNFeatures(xTable->getNumberOfColumns());

    size_t nVectors  = xTable->getNumberOfRows();

    size_t nWeakLearners = 0;               /* Number of weak learners */
    algorithmFPType *alpha = NULL;          /* AdaBoost coefficients */

    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > hTable(
        new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(1, nVectors));
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > wTable(
        new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(1, nVectors));

    /* Get classification labels */
    algorithmFPType *y;             /* Classification labels */
    daal::internal::FeatureMicroTable<algorithmFPType, readOnly, cpu> mtY(yTable.get());
    mtY.getBlockOfColumnValues(0, 0, nVectors, &y);

    NumericTablePtr weakLearnerInputTables[] = {xTable, yTable, wTable};

    /* Run AdaBoost training */
    adaBoostFreundKernel(nVectors, weakLearnerInputTables, hTable, y, r, parameter, &nWeakLearners, &alpha);

    /* Update Ada Boost model with calculated results */
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

    daal::services::daal_free(alpha);
    return;
}

} // namespace daal::algorithms::adaboost::training::internal
}
}
}
} // namespace daal

#endif
