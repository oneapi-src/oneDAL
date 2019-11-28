/* file: boosting_model.cpp */
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
//  Implementation of the base class defining Boosting algorithm model.
//--
*/

#include "algorithms/boosting/boosting_model.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace boosting
{
namespace interface1
{
/** Default constructor. Sets the decision stump as the default weak learner */
Parameter::Parameter() : weakLearnerTraining(new stump::training::Batch<>), weakLearnerPrediction(new stump::prediction::Batch<>) {}

/**
 * Constructs %boosting algorithm parameters from weak learner training and prediction algorithms
 * \param[in] wlTrain       Pointer to the training algorithm of the weak learner
 * \param[in] wlPredict     Pointer to the prediction algorithm of the weak learner
 */
Parameter::Parameter(const SharedPtr<weak_learner::training::Batch> & wlTrain, const SharedPtr<weak_learner::prediction::Batch> & wlPredict)
    : weakLearnerTraining(wlTrain), weakLearnerPrediction(wlPredict)
{}

Status Parameter::check() const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::interface1::Parameter::check());

    DAAL_CHECK_EX(weakLearnerTraining, ErrorNullAuxiliaryAlgorithm, ParameterName, weakLearnerTrainingStr());
    DAAL_CHECK_EX(weakLearnerPrediction, ErrorNullAuxiliaryAlgorithm, ParameterName, weakLearnerPredictionStr());
    return s;
}

Model::Model(size_t nFeatures, services::Status & st) : _nFeatures(nFeatures), _models(new data_management::DataCollection())
{
    if (!_models)
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
}

/**
 *  Returns the number of weak learners constructed during training of the %boosting algorithm
 *  \return The number of weak learners
 */
size_t Model::getNumberOfWeakLearners() const
{
    return _models->size();
}

/**
 *  Returns weak learner model constructed during training of the %boosting algorithm
 *  \param[in] idx  Index of the model in the collection
 *  \return Weak Learner model corresponding to the index idx
 */
weak_learner::ModelPtr Model::getWeakLearnerModel(size_t idx) const
{
    if (idx < _models->size())
    {
        return staticPointerCast<weak_learner::Model, SerializationIface>((*_models)[idx]);
    }
    return weak_learner::ModelPtr();
}

/**
 *  Add weak learner model into the %boosting model
 *  \param[in] model Weak learner model to add into collection
 */
void Model::addWeakLearnerModel(weak_learner::ModelPtr model)
{
    (*_models) << model;
}

void Model::clearWeakLearnerModels()
{
    _models->clear();
}

} // namespace interface1
} // namespace boosting
} // namespace algorithms
} // namespace daal
