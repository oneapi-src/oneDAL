/* file: stump_classification_predict.cpp */
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
//  Implementation of stump algorithm and types methods.
//--
*/

#include "algorithms/stump/stump_classification_predict_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace classification
{
namespace prediction
{
Input::Input() {}
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
 * Returns the input Model object in the prediction stage of the Stump algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          %Input object that corresponds to the given identifier
 */
stump::classification::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<stump::classification::Model, data_management::SerializationIface>(Argument::get(id));
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
 * Sets the input Model object in the prediction stage of the Stump algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const stump::classification::ModelPtr & ptr)
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

    stump::classification::ModelPtr m = get(classifier::prediction::model);
    const size_t modelSplitFeature    = m->getSplitFeature();
    DAAL_CHECK(modelSplitFeature < get(classifier::prediction::data)->getNumberOfColumns(), services::ErrorStumpIncorrectSplitFeature);
    return services::Status();
}

} // namespace prediction
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal
