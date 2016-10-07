/* file: svm_predict.cpp */
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
//  Implementation of svm algorithm and types methods.
//--
*/

#include "svm_predict_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace interface1
{
Input::Input() : classifier::prediction::Input() {}

/**
 * Returns the input Numeric Table object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input NumericTable object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns the input Model object in the prediction stage of the SVM algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          %Input object that corresponds to the given identifier
 */
services::SharedPtr<svm::Model> Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<svm::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the input NumericTable object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input Model object in the prediction stage of the SVM algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const services::SharedPtr<svm::Model> &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    classifier::prediction::Input::check(parameter, method);
    if(this->_errors->size() != 0) { return; }

    services::SharedPtr<svm::Model> m =
        services::staticPointerCast<svm::Model, classifier::Model>(get(classifier::prediction::model));

    services::ErrorCollection errors;
    errors.setCanThrow(false);
    if(!data_management::checkNumericTable(m->getSupportVectors().get(), &errors, supportVectorsStr()))
    {
        this->_errors->add(services::Error::create(services::ErrorModelNotFullInitialized, services::ArgumentName, supportVectorsStr()));
        return;
    }
    if(!data_management::checkNumericTable(get(classifier::prediction::data).get(), this->_errors.get(), dataStr(),
        0, 0, m->getSupportVectors()->getNumberOfColumns())) { return; }
    if(!data_management::checkNumericTable(m->getClassificationCoefficients().get(), &errors, classificationCoefficientsStr()))
    {
        this->_errors->add(services::Error::create(services::ErrorModelNotFullInitialized, services::ArgumentName, classificationCoefficientsStr()));
        return;
    }
}

}// namespace interface1
}// namespace prediction
}// namespace svm
}// namespace algorithms
}// namespace daal
