/* file: adaboost_model.cpp */
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
//  Implementation of class defining Ada Boost model
//--
*/

#include "algorithms/boosting/adaboost_model.h"
#include "algorithms/stump/stump_classification_training_batch.h"
#include "algorithms/stump/stump_classification_predict.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_MULTICLASS_ADABOOST_MODEL_ID);

Parameter::Parameter(size_t nClasses_)
    : classifier::Parameter(nClasses_),
      weakLearnerTraining(new stump::classification::training::Batch<>(nClasses_)),
      weakLearnerPrediction(new stump::classification::prediction::Batch<>(nClasses_)),
      accuracyThreshold(0.0),
      maxIterations(100),
      learningRate(1.0),
      resultsToCompute(computeWeakLearnersErrors)
{}

/**
 * Constructs the AdaBoost parameter structure
 * \param[in] wlTrain       Pointer to the training algorithm of the weak learner
 * \param[in] wlPredict     Pointer to the prediction algorithm of the weak learner
 * \param[in] acc           Accuracy of the AdaBoost training algorithm
 * \param[in] maxIter       Maximal number of iterations of the AdaBoost training algorithm
 */
Parameter::Parameter(SharedPtr<classifier::training::Batch> wlTrain, SharedPtr<classifier::prediction::Batch> wlPredict, double acc, size_t maxIter,
                     double learningRate_, DAAL_UINT64 resultsToCompute_, size_t nClasses_)
    : classifier::Parameter(nClasses_),
      weakLearnerTraining(wlTrain),
      weakLearnerPrediction(wlPredict),
      accuracyThreshold(acc),
      maxIterations(maxIter),
      learningRate(learningRate_),
      resultsToCompute(resultsToCompute_)
{}

Status Parameter::check() const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());
    DAAL_CHECK_EX(nClasses >= 2, ErrorIncorrectParameter, ParameterName, nClassesStr());
    DAAL_CHECK_EX(weakLearnerTraining, ErrorNullAuxiliaryAlgorithm, ParameterName, weakLearnerTrainingStr());
    DAAL_CHECK_EX(nClasses == weakLearnerTraining->parameter().nClasses, ErrorInconsistentNumberOfClasses, ParameterName, weakLearnerTrainingStr());
    DAAL_CHECK_EX(weakLearnerPrediction, ErrorNullAuxiliaryAlgorithm, ParameterName, weakLearnerPredictionStr());
    DAAL_CHECK_EX(nClasses == weakLearnerPrediction->parameter().nClasses, ErrorInconsistentNumberOfClasses, ParameterName,
                  weakLearnerPredictionStr());
    DAAL_CHECK_EX(accuracyThreshold >= 0 && accuracyThreshold < 1, ErrorIncorrectParameter, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations > 0, ErrorIncorrectParameter, ParameterName, maxIterationsStr());
    DAAL_CHECK_EX(learningRate > 0, ErrorIncorrectParameter, ParameterName, learningRateStr());
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

size_t Model::getNumberOfWeakLearners() const
{
    return _models->size();
}

classifier::ModelPtr Model::getWeakLearnerModel(size_t idx) const
{
    if (idx < _models->size())
    {
        return staticPointerCast<classifier::Model, SerializationIface>((*_models)[idx]);
    }
    return classifier::ModelPtr();
}

void Model::addWeakLearnerModel(classifier::ModelPtr model)
{
    (*_models) << model;
}

void Model::clearWeakLearnerModels()
{
    _models->clear();
}

} // namespace adaboost
} // namespace algorithms
} // namespace daal
