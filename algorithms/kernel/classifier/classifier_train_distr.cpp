/* file: classifier_train_distr.cpp */
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
//  Implementation of classifier training methods.
//--
*/

#include "classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace training
{
namespace interface1
{
PartialResult::PartialResult() : daal::algorithms::PartialResult(1) {};

/**
 * Returns the partial result in the training stage of the classification algorithm
 * \param[in] id   Identifier of the partial result, \ref PartialResultId
 * \return         Partial result that corresponds to the given identifier
 */
services::SharedPtr<classifier::Model> PartialResult::get(PartialResultId id) const
{
    return services::staticPointerCast<classifier::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the partial result in the training stage of the classification algorithm
 * \param[in] id    Identifier of the partial result, \ref PartialResultId
 * \param[in] value Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const services::SharedPtr<daal::algorithms::classifier::Model> &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the PartialResult object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
void PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    checkImpl(input, parameter);
}

void PartialResult::checkImpl(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter) const
{
    services::SharedPtr<daal::algorithms::classifier::Model> m = get(partialModel);
    if(!m) { this->_errors->add(services::ErrorNullModel); return; }
}

DistributedInput::DistributedInput() : InputIface(1)
{
    Argument::set(partialModels, data_management::DataCollectionPtr(new data_management::DataCollection()));
}

size_t DistributedInput::getNumberOfFeatures() const
{
    data_management::DataCollectionPtr models = get(classifier::training::partialModels);
    classifier::Model *firstModel =
        static_cast<classifier::Model *>((*models)[0].get());
    return firstModel->getNFeatures();
}

/**
 * Returns input objects of the classification algorithm in the distributed processing mode
 * \param[in] id    Identifier of the input objects
 * \return          Input object that corresponds to the given identifier
 */
data_management::DataCollectionPtr DistributedInput::get(Step2MasterInputId id) const
{
    return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Adds input object on the master node in the training stage of the classification algorithm
 * \param[in] id            Identifier of the input object
 * \param[in] partialResult Pointer to the object
 */
void DistributedInput::add(const Step2MasterInputId &id, const services::SharedPtr<PartialResult> &partialResult)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    collection->push_back(services::staticPointerCast<data_management::SerializationIface, classifier::Model>(
                              partialResult->get(partialModel)));
}

/**
 * Sets input object in the training stage of the classification algorithm
 * \param[in] id   Identifier of the object
 * \param[in] value Pointer to the object
 */
void DistributedInput::set(Step2MasterInputId id, const data_management::DataCollectionPtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks input parameters in the training stage of the classification algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Algorithm method
 */
void DistributedInput::check(const daal::algorithms::Parameter *parameter, int method) const
{
    data_management::DataCollectionPtr spModels = get(partialModels);
    data_management::DataCollection *models = spModels.get();
    if (models == 0) { this->_errors->add(services::ErrorNullModel); return; }

    size_t size = models->size();
    if (size == 0) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

    for (size_t i = 0; i < size; i++)
    {
        classifier::Model *model = (classifier::Model *)((*models)[i].get());
        if (model == 0) { this->_errors->add(services::ErrorNullModel); return; }
    }
}

}
}
}
}
}
