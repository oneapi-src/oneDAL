/* file: logitboost_train_friedman_impl.i */
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
//  Common functions for Logit Boost model training
//--
*/
/*
//
//  REFERENCES
//
//  1. J. Friedman, T. Hastie, R. Tibshirani.
//     Additive logistic regression: a statistical view of boosting,
//     The annals of Statistics, 2000, v28 N2, pp. 337-407
//  2. J. Friedman, T. Hastie, R. Tibshirani.
//     The Elements of Statistical Learning:
//     Data Mining, Inference, and Prediction,
//     Springer, 2001.
//
*/

#ifndef __LOGITBOOST_TRAIN_FRIEDMAN_IMPL_I__
#define __LOGITBOOST_TRAIN_FRIEDMAN_IMPL_I__

#include "threading.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "service_data_utils.h"

#include "logitboost_impl.i"
#include "logitboost_train_friedman_aux.i"

using namespace daal::algorithms::logitboost::internal;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
services::Status LogitBoostTrainKernel<friedman, algorithmFPType, cpu>::compute(const size_t na, NumericTablePtr a[], Model *r, const Parameter *par)
{
    const algorithmFPType zero(0.0);
    const algorithmFPType fp_one(1.0);
    Parameter *parameter = const_cast<Parameter *>(par);
    NumericTablePtr x = a[0];
    NumericTablePtr y = a[1];
    r->setNFeatures(x->getNumberOfColumns());

    algorithmFPType acc = parameter->accuracyThreshold;
    const size_t M = parameter->maxIterations;
    const size_t nc = parameter->nClasses;
    const algorithmFPType thrW = (algorithmFPType)(parameter->weightsDegenerateCasesThreshold);
    const algorithmFPType thrZ = (algorithmFPType)(parameter->responsesDegenerateCasesThreshold);
    const size_t dim = x->getNumberOfColumns();
    const size_t n = x->getNumberOfRows();

    TArray<algorithmFPType, cpu> pred(n * nc);
    TArray<algorithmFPType, cpu> F(n * nc);
    TArray<algorithmFPType, cpu> Fbuf(nc);
    TArray<algorithmFPType, cpu> P(n * nc);

    DAAL_CHECK(pred.get() && F.get() && P.get() && Fbuf.get(), services::ErrorMemoryAllocationFailed);

    HomogenNTPtr wTable(new HomogenNT(1, n));
    HomogenNTPtr zTable(new HomogenNT(1, n));

    algorithmFPType *w = wTable->getArray();
    algorithmFPType *z = zTable->getArray();

    const algorithmFPType inv_n = fp_one / (algorithmFPType)n;
    const algorithmFPType inv_nc = fp_one / (algorithmFPType)nc;

    /* Initialize weights, probs and additive function values.
       Step 1) of the Algorithm 6 from [1] */
    for ( size_t i = 0; i < n; i++ ) { w[i] = inv_n; }
    for ( size_t i = 0; i < n * nc; i++ ) { P[i] = inv_nc; }
    algorithmFPType logL = -algorithmFPType(n)*daal::internal::Math<algorithmFPType, cpu>::sLog(inv_nc);
    algorithmFPType accCur = daal::data_feature_utils::internal::MaxVal<algorithmFPType, cpu>::get();

    for (size_t i = 0; i < n * nc; i++)
    {
        F[i] = zero;
    }

    services::Status s;
    ReadColumns<int, cpu> yCols(*y, 0, 0, n);
    DAAL_CHECK_BLOCK_STATUS(yCols);
    const int *y_label = yCols.get();
    DAAL_ASSERT(y_label);

    services::SharedPtr<weak_learner::training::Batch>   learnerTrain   = parameter->weakLearnerTraining;
    learnerTrain->input.set(classifier::training::data,    x);
    learnerTrain->input.set(classifier::training::labels,  zTable);
    learnerTrain->input.set(classifier::training::weights, wTable);

    services::SharedPtr<weak_learner::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction;
    learnerPredict->inputBase->set(classifier::prediction::data, x);

    /* Clear the collection of weak learners models in the boosting model */
    r->clearWeakLearnerModels();

    /* Repeat for m = 0, 1, ..., M-1
       Step 2) of the Algorithm 6 from [1] */
    for ( size_t m = 0; m < M; m++ )
    {
        /* Repeat for j = 0, 1, ..., nk-1
           Step 2.a) of the Algorithm 6 from [1] */
        for ( size_t j = 0; j < nc; j++ )
        {
            initWZ<algorithmFPType, cpu>(n, nc, j, y_label, P.get(), thrW, w, thrZ, z);

            learnerTrain->resetResult();
            DAAL_CHECK_STATUS(s, learnerTrain->computeNoThrow());

            classifier::training::ResultPtr trainingRes = learnerTrain->getResult();
            weak_learner::ModelPtr learnerModel =
                services::staticPointerCast<weak_learner::Model, classifier::Model>(trainingRes->get(classifier::training::model));

            /* Add new model to the collection of the boosting algorithm models */
            r->addWeakLearnerModel(learnerModel);

            learnerPredict->inputBase->set(classifier::prediction::model, learnerModel);
            HomogenNTPtr predTable(new HomogenNT(pred.get() + j * n, 1, n));

            classifier::prediction::ResultPtr predictionRes(new classifier::prediction::Result());
            predictionRes->set(classifier::prediction::prediction, predTable);
            s = learnerPredict->setResult(predictionRes);
            if(s)
                s = learnerPredict->computeNoThrow();
            if(!s)
                return s;
            }

        /* Update additive function's values
           Step 2.b) of the Algorithm 6 from [1] */
        /* i-row contains Fi() for all classes in i-th point x */
        UpdateF<algorithmFPType, cpu>(dim, n, nc, pred.get(), F.get());

        /* Update probabilities
           Step 2.c) of the Algorithm 6 from [1] */
        UpdateP<algorithmFPType, cpu>(nc, n, F.get(), P.get(), Fbuf.get());

        /* Calculate model accuracy */
        calculateAccuracy<algorithmFPType, cpu>(n, nc, y_label, P.get(), logL, accCur);

        if (accCur < acc)
        {
            r->setIterations( m + 1 );
            break;
        }
    }
    return s;
}

} // namepsace internal
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
