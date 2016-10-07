/* file: boosting_predict_impl.i */
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
//  Implementation of common method for boosting prediction algorithms.
//--
*/

#ifndef __BOOSTING_PREDICT_IMPL_I__
#define __BOOSTING_PREDICT_IMPL_I__

#include "service_memory.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace boosting
{
namespace prediction
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
void BoostingPredictKernel<algorithmFPType, cpu>::compute(NumericTablePtr xTable, const Model *m, size_t nWeakLearners,
                 const algorithmFPType *alpha, algorithmFPType *r, const Parameter *par)
{
    size_t nVectors  = xTable->getNumberOfRows();
    Model *boostModel = const_cast<Model *>(m);
    Parameter *parameter = const_cast<Parameter *>(par);

    const algorithmFPType zero = (algorithmFPType)0.0;
    const algorithmFPType one  = (algorithmFPType)1.0;

    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > rWeakTable(
        new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(1, nVectors));
    algorithmFPType *rWeak = rWeakTable->getArray();

    services::SharedPtr<weak_learner::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction;
    learnerPredict->inputBase->set(classifier::prediction::data, xTable);

    services::SharedPtr<classifier::prediction::Result> predictionRes(new classifier::prediction::Result());
    predictionRes->set(classifier::prediction::prediction, rWeakTable);
    learnerPredict->setResult(predictionRes);

    /* Initialize array of prediction results */
    for (size_t j = 0; j < nVectors; j++)
    {
        r[j] = zero;
    }

    for (size_t i = 0; i < nWeakLearners; i++)
    {
        /* Get  weak learner's classification results */
        services::SharedPtr<weak_learner::Model> learnerModel = boostModel->getWeakLearnerModel(i);

        learnerPredict->inputBase->set(classifier::prediction::model, learnerModel);
        learnerPredict->computeNoThrow();
        if(learnerPredict->getErrors()->size() != 0) {this->_errors->add(learnerPredict->getErrors()->getErrors()); return;}

        /* Update boosting classification results */
        for (size_t j = 0; j < nVectors; j++)
        {
            algorithmFPType p = ((rWeak[j] > zero) ? one : -one);
            r[j] += p * alpha[i];
        }
    }
}

}
}
}
}
}

#endif
