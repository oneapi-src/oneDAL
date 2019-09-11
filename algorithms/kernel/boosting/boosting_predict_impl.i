/* file: boosting_predict_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
services::Status BoostingPredictKernel<algorithmFPType, cpu>::compute(const NumericTablePtr& xTable,
    const Model *m, size_t nWeakLearners, const algorithmFPType *alpha, algorithmFPType *r, const Parameter *par)
{
    const size_t nVectors  = xTable->getNumberOfRows();
    Model *boostModel = const_cast<Model *>(m);
    Parameter *parameter = const_cast<Parameter *>(par);

    services::Status s;
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > rWeakTable = daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(1, nVectors, &s);
    DAAL_CHECK_STATUS_VAR(s);
    const algorithmFPType *rWeak = rWeakTable->getArray();

    services::SharedPtr<weak_learner::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction->clone();
    classifier::prediction::Input *learnerInput = learnerPredict->getInput();
    DAAL_CHECK(learnerInput, services::ErrorNullInput);
    learnerInput->set(classifier::prediction::data, xTable);

    classifier::prediction::ResultPtr predictionRes(new classifier::prediction::Result());
    predictionRes->set(classifier::prediction::prediction, rWeakTable);
    DAAL_CHECK_STATUS(s, learnerPredict->setResult(predictionRes));

    const algorithmFPType zero = (algorithmFPType)0.0;

    /* Initialize array of prediction results */
    for (size_t j = 0; j < nVectors; j++)
    {
        r[j] = zero;
    }

    const algorithmFPType one = (algorithmFPType)1.0;
    for(size_t i = 0; i < nWeakLearners; i++)
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

}
}
}
}
}

#endif
