/* file: svm_predict.cpp */
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
//  Implementation of svm algorithm and types methods.
//--
*/

#include "algorithms/svm/svm_predict_types.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
using namespace daal::data_management;
using namespace daal::services;

Input::Input() : classifier::prediction::Input() {}
Input::Input(const Input & other) : classifier::prediction::Input(other) {}

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
svm::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<svm::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the input NumericTable object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input Model object in the prediction stage of the SVM algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const svm::ModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::prediction::Input::check(parameter, method));

    svm::ModelPtr m = services::staticPointerCast<svm::Model, classifier::Model>(get(classifier::prediction::model));

    s = data_management::checkNumericTable(m->getSupportVectors().get(), supportVectorsStr());
    if (!s) return Status(services::Error::create(services::ErrorModelNotFullInitialized, services::ArgumentName, supportVectorsStr()));

    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(classifier::prediction::data).get(), dataStr(), 0, 0,
                                                            m->getSupportVectors()->getNumberOfColumns()));
    s |= data_management::checkNumericTable(m->getClassificationCoefficients().get(), classificationCoefficientsStr());
    if (!s) return Status(services::Error::create(services::ErrorModelNotFullInitialized, services::ArgumentName, classificationCoefficientsStr()));
    return s;
}

} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal
