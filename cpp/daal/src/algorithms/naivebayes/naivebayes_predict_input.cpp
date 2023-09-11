/* file: naivebayes_predict_input.cpp */
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
//  Implementation of multinomial naive bayes algorithm and types methods.
//--
*/

#include "algorithms/naive_bayes/multinomial_naive_bayes_predict_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace prediction
{
Input::Input() {}
Input::Input(const Input & other) : classifier::prediction::Input(other) {}

/**
 * Returns the input Numeric Table object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input NumericTable object
 * \return          Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns the input Model object in the prediction stage of the multinomial naive Bayes algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          Input object that corresponds to the given identifier
 */
multinomial_naive_bayes::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<multinomial_naive_bayes::Model, data_management::SerializationIface>(Argument::get(id));
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
 * Sets the input Model object in the prediction stage of the multinomial naive Bayes algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const multinomial_naive_bayes::ModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::prediction::Input::check(parameter, method));

    if (method == fastCSR)
    {
        int expectedLayouts = (int)NumericTableIface::csrArray;
        DAAL_CHECK_STATUS(s, checkNumericTable(get(classifier::prediction::data).get(), dataStr(), 0, expectedLayouts));
    }

    ModelPtr inputModel = get(classifier::prediction::model);
    DAAL_CHECK(inputModel, ErrorNullModel);

    DAAL_CHECK(inputModel->getLogP(), services::ErrorModelNotFullInitialized);
    DAAL_CHECK(inputModel->getLogTheta(), services::ErrorModelNotFullInitialized);
    DAAL_CHECK(inputModel->getAuxTable(), services::ErrorModelNotFullInitialized);

    size_t nClasses = 0;

    const multinomial_naive_bayes::Parameter * algPar2 = dynamic_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    if (algPar2) nClasses = algPar2->nClasses;
    DAAL_CHECK_EX(nClasses > 0, ErrorNullParameterNotSupported, ArgumentName, nClassesStr());

    DAAL_CHECK(inputModel->getLogP()->getNumberOfRows() == nClasses, ErrorNaiveBayesIncorrectModel);

    return s;
}

} // namespace prediction
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
