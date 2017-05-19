/* file: multiclassclassifier_train.cpp */
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
//  Implementation of multi class classifier algorithm and types methods.
//--
*/

#include "multi_class_classifier_train_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{

namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_MULTI_CLASS_CLASSIFIER_MODEL_ID);

Model::Model(const ParameterBase *par) : _modelsArray(NULL), _models(new data_management::DataCollection(par->nClasses * (par->nClasses - 1) / 2))
{
}

Model::Model() : _modelsArray(NULL), _models(new data_management::DataCollection())
{
}

Model::~Model()
{
    delete[] _modelsArray;
}

classifier::ModelPtr *Model::getTwoClassClassifierModels()
{
    if(!_modelsArray)
    {
        _modelsArray = new classifier::ModelPtr[_models->size()];
        for(size_t i = 0; i < _models->size(); i++)
        {
            _modelsArray[i] = services::staticPointerCast<classifier::Model, data_management::SerializationIface>((*_models)[i]);
        }
    }
    return _modelsArray;
}

void Model::setTwoClassClassifierModel(size_t idx, const classifier::ModelPtr& model)
{
    (*_models)[idx] = model;
}

classifier::ModelPtr Model::getTwoClassClassifierModel(size_t idx) const
{
    if(idx < _models->size())
    {
        return services::staticPointerCast<classifier::Model, data_management::SerializationIface>((*_models)[idx]);
    }
    return classifier::ModelPtr();
}

services::Status ParameterBase::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());
    DAAL_CHECK_EX(training.get(), services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, trainingStr());
    DAAL_CHECK_EX(prediction.get(), services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, predictionStr());
    return s;
}

services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, ParameterBase::check());
    DAAL_CHECK_EX((accuracyThreshold > 0) && (accuracyThreshold < 1), services::ErrorIncorrectParameter, services::ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations, services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr());
    return s;
}

}

namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_MULTICLASS_CLASSIFIER_RESULT_ID);
Result::Result() {}

/**
 * Returns the model trained with the Multi class classifier algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the Multi class classifier algorithm
 */
daal::algorithms::multi_class_classifier::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::multi_class_classifier::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Checks the correctness of the Result object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, checkImpl(input, parameter));

    const Parameter *par = static_cast<const Parameter *>(parameter);
    daal::algorithms::multi_class_classifier::ModelPtr m = get(classifier::training::model);
    if(m->getNumberOfTwoClassClassifierModels() == 0)
    {
        return services::Status(services::ErrorModelNotFullInitialized);
    }
    if(m->getNumberOfTwoClassClassifierModels() != par->nClasses * (par->nClasses - 1) / 2)
    {
        return services::Status(services::ErrorModelNotFullInitialized);
    }
    return s;
}

}// namespace interface1
}// namespace training
}// namespace multi_class_classifier
}// namespace algorithms
}// namespace daal
