/* file: brownboost_predict_impl.i */
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
//  Implementation of Fast method for Brown Boost prediction algorithm.
//--
*/

#ifndef __BROWNBOOST_PREDICT_IMPL_I__
#define __BROWNBOOST_PREDICT_IMPL_I__

#include "src/externals/service_math.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services::internal;

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status BrownBoostPredictKernel<method, algorithmFPType, cpu>::computeImpl(const NumericTablePtr & xTable, const Model * m,
                                                                                    size_t nWeakLearners, const algorithmFPType * alpha,
                                                                                    algorithmFPType * r, const Parameter * par)
{
    const size_t nVectors       = xTable->getNumberOfRows();
    const Model * boostModel    = const_cast<Model *>(m);
    const Parameter * parameter = const_cast<Parameter *>(par);

    services::Status s;
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > rWeakTable =
        daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(1, nVectors, &s);
    DAAL_CHECK_STATUS_VAR(s);
    const algorithmFPType * rWeak = rWeakTable->getArray();

    services::SharedPtr<classifier::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction->clone();
    classifier::prediction::Input * learnerInput                      = learnerPredict->getInput();
    DAAL_CHECK(learnerInput, services::ErrorNullInput);
    learnerInput->set(classifier::prediction::data, xTable);

    classifier::prediction::ResultPtr predictionRes(new classifier::prediction::Result());
    DAAL_CHECK_MALLOC(predictionRes.get())
    predictionRes->set(classifier::prediction::prediction, rWeakTable);
    DAAL_CHECK_STATUS(s, learnerPredict->setResult(predictionRes));

    const algorithmFPType zero = (algorithmFPType)0.0;
    const algorithmFPType one  = (algorithmFPType)1.0;

    /* Initialize array of prediction results */
    service_memset<algorithmFPType, cpu>(r, zero, nVectors);

    for (size_t i = 0; i < nWeakLearners; i++)
    {
        /* Get  weak learner's classification results */
        classifier::ModelPtr learnerModel = boostModel->getWeakLearnerModel(i);

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

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status BrownBoostPredictKernel<method, algorithmFPType, cpu>::compute(const NumericTablePtr & xTable, const Model * m,
                                                                                const NumericTablePtr & rTable, const Parameter * par)
{
    const size_t nVectors      = xTable->getNumberOfRows();
    Model * boostModel         = const_cast<Model *>(m);
    const size_t nWeakLearners = boostModel->getNumberOfWeakLearners();

    services::Status s;
    WriteOnlyColumns<algorithmFPType, cpu> mtR(*rTable, 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * r = mtR.get();
    DAAL_ASSERT(r);

    {
        ReadColumns<algorithmFPType, cpu> mtAlpha(*boostModel->getAlpha(), 0, 0, nWeakLearners);
        DAAL_CHECK_BLOCK_STATUS(mtAlpha);
        DAAL_ASSERT(mtAlpha.get());
        DAAL_CHECK_STATUS(s, this->computeImpl(xTable, m, nWeakLearners, mtAlpha.get(), r, par));
    }

    Parameter * parameter       = const_cast<Parameter *>(par);
    const algorithmFPType error = parameter->accuracyThreshold;
    const algorithmFPType zero  = (algorithmFPType)0.0;
    if (error != zero)
    {
        algorithmFPType sqrtC    = daal::internal::MathInst<algorithmFPType, cpu>::sErfInv(algorithmFPType(1.0) - error);
        algorithmFPType invSqrtC = algorithmFPType(1.0) / sqrtC;
        for (size_t j = 0; j < nVectors; j++)
        {
            r[j] *= invSqrtC;
        }
    }
    daal::internal::MathInst<algorithmFPType, cpu>::vErf(nVectors, r, r);
    return s;
}

} // namespace internal
} // namespace prediction
} // namespace brownboost
} // namespace algorithms
} // namespace daal

#endif
