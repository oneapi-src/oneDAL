/* file: classifier_train.cpp */
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
InputIface::InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
Input::Input() : InputIface(3) {}

size_t Input::getNumberOfFeatures() const
{
    return get(classifier::training::data)->getNumberOfColumns();
}

/**
 * Returns the input object in the training stage of the classification algorithm
 * \param[in] id   Identifier of the input object, \ref InputId
 * \return         Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the input object in the training stage of the classification algorithm
 * \param[in] id    Identifier of the input object, \ref InputId
 * \param[in] value Pointer to the input object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    checkImpl(parameter);
}

void Input::checkImpl(const daal::algorithms::Parameter *parameter) const
{
    if (parameter != NULL)
    {
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        if (algParameter->nClasses < 2)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nClassesStr()));
            return;
        }
    }

    data_management::NumericTablePtr dataTable = get(data);
    if(!data_management::checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }

    size_t nRows = dataTable->getNumberOfRows();

    data_management::NumericTablePtr labelsTable = get(labels);
    if(!data_management::checkNumericTable(labelsTable.get(), this->_errors.get(), labelsStr(), 0, 0, 1, nRows)) { return; }

    data_management::NumericTablePtr weightsTable = get(weights);
    if(weightsTable)
    {
        if(!data_management::checkNumericTable(weightsTable.get(), this->_errors.get(), weightsStr(), 0, 0, 1, nRows)) { return; }
    }
}

Result::Result() : daal::algorithms::Result(1) {}

/**
 * Returns the model trained with the classification algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \return          Model trained with the classification algorithm
 */
services::SharedPtr<daal::algorithms::classifier::Model> Result::get(ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::classifier::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the training stage of the classification algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \param[in] value Pointer to the training result
 */
void Result::set(ResultId id, const services::SharedPtr<daal::algorithms::classifier::Model> &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
           int method) const
{
    checkImpl(input, parameter);
}

void Result::checkImpl(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter) const
{
    services::SharedPtr<daal::algorithms::classifier::Model> m = get(model);
    if(!m) { this->_errors->add(services::ErrorNullModel); return; }
}

}
}
}
}
}
