/* file: adaboost_model_v1.cpp */
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
//  Implementation of class defining Ada Boost model
//--
*/

#include "algorithms/boosting/adaboost_model.h"
#include "algorithms/stump/stump_classification_training_batch.h"
#include "algorithms/stump/stump_classification_predict.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_ADABOOST_MODEL_ID);

Parameter::Parameter() : boosting::Parameter(), accuracyThreshold(0.0), maxIterations(10) {}

/**
 * Constructs the AdaBoost parameter structure
 * \param[in] wlTrain       Pointer to the training algorithm of the weak learner
 * \param[in] wlPredict     Pointer to the prediction algorithm of the weak learner
 * \param[in] acc           Accuracy of the AdaBoost training algorithm
 * \param[in] maxIter       Maximal number of iterations of the AdaBoost training algorithm
 */
Parameter::Parameter(SharedPtr<weak_learner::training::Batch> wlTrain, SharedPtr<weak_learner::prediction::Batch> wlPredict, double acc,
                     size_t maxIter)
    : boosting::Parameter(wlTrain, wlPredict), accuracyThreshold(acc), maxIterations(maxIter)
{}

Status Parameter::check() const
{
    Status s;
    DAAL_CHECK_STATUS(s, boosting::Parameter::check());
    DAAL_CHECK_EX(accuracyThreshold >= 0 && accuracyThreshold < 1, ErrorIncorrectParameter, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations > 0, ErrorIncorrectParameter, ParameterName, maxIterationsStr());
    return s;
}

/**
 *  Returns a pointer to the array of weights of weak learners constructed
 *  during training of the AdaBoost algorithm.
 *  The size of the array equals the number of weak learners
 *  \return Array of weights of weak learners.
 */
NumericTablePtr Model::getAlpha() const
{
    return _alpha;
}

} // namespace interface1
} // namespace adaboost
} // namespace algorithms
} // namespace daal
