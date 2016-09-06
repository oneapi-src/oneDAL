/* file: implicit_als_train_result.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "implicit_als_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace interface1
{
/** Default constructor */
Result::Result() : daal::algorithms::Result(1) {}

/**
 * Returns the result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
services::SharedPtr<daal::algorithms::implicit_als::Model> Result::get(ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::implicit_als::Model,
           data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const services::SharedPtr<daal::algorithms::implicit_als::Model> &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the implicit ALS training algorithm
 * \param[in] input       %Input object for the algorithm
 * \param[in] parameter   %Parameter of the algorithm
 * \param[in] method      Computation method of the algorithm
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    NumericTablePtr dataTable = algInput->get(data);
    size_t nUsers = dataTable->getNumberOfRows();
    size_t nItems = dataTable->getNumberOfColumns();

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors = algParameter->nFactors;

    ModelPtr trainedModel = get(model);
    DAAL_CHECK(trainedModel, ErrorNullModel);

    int unexpectedLayouts = (int)packed_mask;
    if(!checkNumericTable(trainedModel->getUsersFactors().get(), this->_errors.get(), usersFactorsStr(), unexpectedLayouts, 0, nFactors, nUsers)) { return; }
    if(!checkNumericTable(trainedModel->getItemsFactors().get(), this->_errors.get(), itemsFactorsStr(), unexpectedLayouts, 0, nFactors, nItems)) { return; }
}

}// namespace interface1
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
