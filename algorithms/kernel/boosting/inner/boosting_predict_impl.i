/* file: boosting_predict_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
services::Status BoostingPredictKernel<algorithmFPType, cpu>::compute(const NumericTablePtr & xTable, const Model * m, size_t nWeakLearners,
                                                                      const algorithmFPType * alpha, algorithmFPType * r, const Parameter * par)
{
    const size_t nVectors = xTable->getNumberOfRows();
    Model * boostModel    = const_cast<Model *>(m);
    Parameter * parameter = const_cast<Parameter *>(par);

    services::Status s;
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > rWeakTable =
        daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(1, nVectors, &s);
    DAAL_CHECK_STATUS_VAR(s);
    const algorithmFPType * rWeak = rWeakTable->getArray();

    services::SharedPtr<weak_learner::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction->clone();
    classifier::prediction::interface1::Input * learnerInput            = learnerPredict->getInput();
    DAAL_CHECK(learnerInput, services::ErrorNullInput);
    learnerInput->set(classifier::prediction::data, xTable);

    classifier::prediction::interface1::ResultPtr predictionRes(new classifier::prediction::interface1::Result());
    predictionRes->set(classifier::prediction::prediction, rWeakTable);
    DAAL_CHECK_STATUS(s, learnerPredict->setResult(predictionRes));

    const algorithmFPType zero = (algorithmFPType)0.0;

    /* Initialize array of prediction results */
    for (size_t j = 0; j < nVectors; j++)
    {
        r[j] = zero;
    }

    const algorithmFPType one = (algorithmFPType)1.0;
    for (size_t i = 0; i < nWeakLearners; i++)
    {
        /* Get  weak learner's classification results */
        weak_learner::ModelPtr learnerModel = boostModel->getWeakLearnerModel(i);

        learnerInput->set(classifier::prediction::model, learnerModel);
        DAAL_CHECK_STATUS(s, learnerPredict->computeNoThrow());

        /* Update boosting classification results */
        for (size_t j = 0; j < nVectors; j++)
        {
            algorithmFPType p = ((rWeak[j] > zero) ? one : -one);
            r[j] += p * alpha[i];
        }
    }
    return s;
}

} // namespace internal
} // namespace prediction
} // namespace boosting
} // namespace algorithms
} // namespace daal

#endif
