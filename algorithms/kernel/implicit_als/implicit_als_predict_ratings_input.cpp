/* file: implicit_als_predict_ratings_input.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "implicit_als_predict_ratings_types.h"
#include "daal_strings.h"

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
namespace interface1
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
void Input::set(ModelInputId id, const ModelPtr &ptr)
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
    if(trainedModel)
    {
        data_management::NumericTablePtr factors = trainedModel->getUsersFactors();
        if(factors)
            return factors->getNumberOfRows();
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
    if(trainedModel)
    {
        data_management::NumericTablePtr factors = trainedModel->getItemsFactors();
        if(factors)
            return factors->getNumberOfRows();
    }
    return 0;
}

services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    DAAL_CHECK(parameter, ErrorNullParameterNotSupported);
    const Parameter *alsParameter = static_cast<const Parameter *>(parameter);
    const size_t nFactors = alsParameter->nFactors;

    ModelPtr trainedModel = get(model);
    DAAL_CHECK(trainedModel, ErrorNullModel);

    const int unexpectedLayouts = (int)packed_mask;
    services::Status s = checkNumericTable(trainedModel->getUsersFactors().get(), usersFactorsStr(), unexpectedLayouts, 0, nFactors);
    s |= checkNumericTable(trainedModel->getItemsFactors().get(), itemsFactorsStr(), unexpectedLayouts, 0, nFactors);
    return s;
}

}// namespace interface1
}// namespace ratings
}// namespace prediction
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
