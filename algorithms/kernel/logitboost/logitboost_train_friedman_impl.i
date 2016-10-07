/* file: logitboost_train_friedman_impl.i */
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
void LogitBoostTrainKernel<friedman, algorithmFPType, cpu>::compute(const size_t na, NumericTablePtr a[],
            Model *r, const Parameter *par)
{
    size_t dim, n, nc, M;
    const algorithmFPType zero = 0.0;
    algorithmFPType acc, accCur;
    algorithmFPType thrW, thrZ;
    algorithmFPType logL = 0.0;

    algorithmFPType *pred, *F, *P, *Fbuf;
    algorithmFPType inv_n, inv_nc;
    algorithmFPType fp_one = (algorithmFPType)1.0;
    Parameter *parameter = const_cast<Parameter *>(par);
    NumericTablePtr x = a[0];
    NumericTablePtr y = a[1];
    r->setNFeatures(x->getNumberOfColumns());

    acc = parameter->accuracyThreshold;
    M   = parameter->maxIterations;
    nc  = parameter->nClasses;
    thrW = (algorithmFPType)(parameter->weightsDegenerateCasesThreshold);
    thrZ = (algorithmFPType)(parameter->responsesDegenerateCasesThreshold);
    dim = x->getNumberOfColumns();
    n   = x->getNumberOfRows();

    pred = (algorithmFPType *) daal::services::daal_malloc (n * nc * sizeof(algorithmFPType));
    F    = (algorithmFPType *) daal::services::daal_malloc (n * nc * sizeof(algorithmFPType));
    Fbuf = (algorithmFPType *) daal::services::daal_malloc (    nc * sizeof(algorithmFPType));
    P    = (algorithmFPType *) daal::services::daal_malloc (n * nc * sizeof(algorithmFPType));

    if (!pred || !F || !P || !Fbuf)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed); return;
    }
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > wTable(
        new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(1, n));
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > zTable(
        new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(1, n));
    algorithmFPType *w = wTable->getArray();
    algorithmFPType *z = zTable->getArray();

    inv_n  = fp_one / (algorithmFPType)n;
    inv_nc = fp_one / (algorithmFPType)nc;

    /* Initialize weights, probs and additive function values.
       Step 1) of the Algorithm 6 from [1] */
    for ( size_t i = 0; i < n; i++ ) { w[i] = inv_n; }
    for ( size_t i = 0; i < n * nc; i++ ) { P[i] = inv_nc; }
    for ( size_t i = 0; i < n; i++)
    {
        logL -= daal::internal::Math<algorithmFPType,cpu>::sLog(inv_nc);
    }
    accCur = daal::data_feature_utils::internal::MaxVal<algorithmFPType, cpu>::get();

    for (size_t i = 0; i < n * nc; i++)
    {
        F[i] = zero;
    }

    int *y_label;
    BlockDescriptor<int> block;
    y->getBlockOfColumnValues( 0, 0, n, readOnly, block );
    y_label = block.getBlockPtr();

    services::SharedPtr<weak_learner::training::Batch>   learnerTrain   = parameter->weakLearnerTraining;
    learnerTrain->getErrors()->setCanThrow(false);
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
            initWZ<algorithmFPType, cpu>(n, nc, j, y_label, P, thrW, w, thrZ, z);

            learnerTrain->resetResult();
            learnerTrain->computeNoThrow();
            if(learnerTrain->getErrors()->size() != 0)
            {
                daal::services::daal_free (pred);
                daal::services::daal_free (F);
                daal::services::daal_free (Fbuf);
                daal::services::daal_free (P);
                this->_errors->add(learnerTrain->getErrors()->getErrors());
                return;
            }

            services::SharedPtr<classifier::training::Result> trainingRes = learnerTrain->getResult();
            services::SharedPtr<weak_learner::Model> learnerModel =
                services::staticPointerCast<weak_learner::Model, classifier::Model>(trainingRes->get(classifier::training::model));

            /* Add new model to the collection of the boosting algorithm models */
            r->addWeakLearnerModel(learnerModel);

            learnerPredict->inputBase->set(classifier::prediction::model, learnerModel);
            services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > predTable(
                new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(pred + j * n, 1, n));
            services::SharedPtr<classifier::prediction::Result> predictionRes(new classifier::prediction::Result());
            predictionRes->set(classifier::prediction::prediction, predTable);
            learnerPredict->setResult(predictionRes);
            learnerPredict->computeNoThrow();
            if(learnerPredict->getErrors()->size() != 0)
            {
                daal::services::daal_free (pred);
                daal::services::daal_free (F);
                daal::services::daal_free (Fbuf);
                daal::services::daal_free (P);
                this->_errors->add(learnerTrain->getErrors()->getErrors());
                return;
            }
        }

        /* Update additive function's values
           Step 2.b) of the Algorithm 6 from [1] */
        /* i-row contains Fi() for all classes in i-th point x */
        UpdateF<algorithmFPType, cpu>( dim, n, nc, pred, F );

        /* Update probabilities
           Step 2.c) of the Algorithm 6 from [1] */
        UpdateP<algorithmFPType, cpu>( nc, n, F, P, Fbuf );

        /* Calculate model accuracy */
        calculateAccuracy<algorithmFPType, cpu>( n, nc, y_label, P, &logL, &accCur );

        if (accCur < acc)
        {
            r->setIterations( m + 1 );
            break;
        }
    }

    y->releaseBlockOfColumnValues( block );

    daal::services::daal_free (pred);
    daal::services::daal_free (F);
    daal::services::daal_free (Fbuf);
    daal::services::daal_free (P);
}

} // namepsace internal
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
