/* file: implicit_als_predict_ratings_input.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_predict_ratings_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
Input::Input() : InputIface(lastModelInputId + 1) {}

/**
 * Returns an input Model object for the rating prediction stage of the implicit ALS algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          Input object that corresponds to the given identifier
 */
ModelPtr Input::get(ModelInputId id) const
{
    return services::staticPointerCast<Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input Model object for the rating prediction stage of the implicit ALS algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(ModelInputId id, const ModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the number of rows in the input numeric table
 * \return Number of rows in the input numeric table
 */
size_t Input::getNumberOfUsers() const
{
    ModelPtr trainedModel = get(model);
    if (trainedModel)
    {
        data_management::NumericTablePtr factors = trainedModel->getUsersFactors();
        if (factors) return factors->getNumberOfRows();
    }
    return 0;
}

/**
 * Returns the number of columns in the input numeric table
 * \return Number of columns in the input numeric table
 */
size_t Input::getNumberOfItems() const
{
    ModelPtr trainedModel = get(model);
    if (trainedModel)
    {
        data_management::NumericTablePtr factors = trainedModel->getItemsFactors();
        if (factors) return factors->getNumberOfRows();
    }
    return 0;
}

services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DAAL_CHECK(parameter, ErrorNullParameterNotSupported);
    const Parameter * alsParameter = static_cast<const Parameter *>(parameter);
    const size_t nFactors          = alsParameter->nFactors;

    ModelPtr trainedModel = get(model);
    DAAL_CHECK(trainedModel, ErrorNullModel);

    const int unexpectedLayouts = (int)packed_mask;
    services::Status s          = checkNumericTable(trainedModel->getUsersFactors().get(), usersFactorsStr(), unexpectedLayouts, 0, nFactors);
    s |= checkNumericTable(trainedModel->getItemsFactors().get(), itemsFactorsStr(), unexpectedLayouts, 0, nFactors);
    return s;
}

} // namespace ratings
} // namespace prediction
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
