/* file: boosting_model.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of the base class defining Boosting algorithm model.
//--
*/

#include "algorithms/boosting/adaboost_model.h"
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
Parameter::Parameter() :
    weakLearnerTraining(new stump::training::Batch<>),
    weakLearnerPrediction(new stump::prediction::Batch<>) {}

/**
 * Constructs %boosting algorithm parameters from weak learner training and prediction algorithms
 * \param[in] wlTrain       Pointer to the training algorithm of the weak learner
 * \param[in] wlPredict     Pointer to the prediction algorithm of the weak learner
 */
Parameter::Parameter(const SharedPtr<weak_learner::training::Batch>& wlTrain,
    const SharedPtr<weak_learner::prediction::Batch>& wlPredict) :
    weakLearnerTraining(wlTrain), weakLearnerPrediction(wlPredict) {}

Status Parameter::check() const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());

    DAAL_CHECK_EX(weakLearnerTraining, ErrorNullAuxiliaryAlgorithm, ParameterName, weakLearnerTrainingStr());
    DAAL_CHECK_EX(weakLearnerPrediction, ErrorNullAuxiliaryAlgorithm, ParameterName, weakLearnerPredictionStr());
    return s;
}

Model::Model(size_t nFeatures, services::Status &st) :
    _nFeatures(nFeatures),
    _models(new data_management::DataCollection())
{
    if (!_models) { st.add(services::ErrorMemoryAllocationFailed); }
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
    if(idx < _models->size())
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
