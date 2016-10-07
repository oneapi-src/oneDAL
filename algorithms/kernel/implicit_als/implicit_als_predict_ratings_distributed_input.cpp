/* file: implicit_als_predict_ratings_distributed_input.cpp */
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

#include "implicit_als_predict_ratings_types.h"

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

DistributedInput<step1Local>::DistributedInput() : InputIface(2) {}

/**
 * Returns an input object for the rating prediction stage of the implicit ALS algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
services::SharedPtr<PartialModel> DistributedInput<step1Local>::get(PartialModelInputId id) const
{
    return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the rating prediction stage of the implicit ALS algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the new input object value
 */
void DistributedInput<step1Local>::set(PartialModelInputId id, const services::SharedPtr<PartialModel> &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the number of rows in the input numeric table
 * \return Number of rows in the input numeric table
 */
size_t DistributedInput<step1Local>::getNumberOfUsers() const
{
    services::SharedPtr<PartialModel> usersModel = get(usersPartialModel);
    if (!usersModel) { this->_errors->add(services::ErrorNullPartialModel); return 0; }

    data_management::NumericTablePtr factors = usersModel->getFactors();
    if (!factors) { this->_errors->add(services::ErrorNullInputNumericTable); return 0; }

    return factors->getNumberOfRows();
}

/**
 * Returns the number of columns in the input numeric table
 * \return Number of columns in the input numeric table
 */
size_t DistributedInput<step1Local>::getNumberOfItems() const
{
    services::SharedPtr<PartialModel> itemsModel = get(itemsPartialModel);
    if (!itemsModel) { this->_errors->add(services::ErrorNullPartialModel); return 0; }

    data_management::NumericTablePtr factors = itemsModel->getFactors();
    if (!factors) { this->_errors->add(services::ErrorNullInputNumericTable); return 0; }

    return factors->getNumberOfRows();
}

/**
 * Checks the parameters of the rating prediction stage of the implicit ALS algorithm
 * \param[in] parameter     Algorithm %parameter
 * \param[in] method        Computation method for the algorithm
 */
void DistributedInput<step1Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors = algParameter->nFactors;

    PartialModelPtr usersModel = get(usersPartialModel);
    PartialModelPtr itemsModel = get(itemsPartialModel);
    DAAL_CHECK(usersModel, ErrorNullPartialModel);
    DAAL_CHECK(itemsModel, ErrorNullPartialModel);

    int unexpectedLayouts = (int)NumericTableIface::upperPackedTriangularMatrix |
        (int)NumericTableIface::lowerPackedTriangularMatrix |
        (int)NumericTableIface::upperPackedSymmetricMatrix |
        (int)NumericTableIface::lowerPackedSymmetricMatrix;

    int unexpectedLayoutsIndices = unexpectedLayouts | (int)NumericTableIface::csrArray;

    if(!checkNumericTable(usersModel->getFactors().get(), this->_errors.get(), usersFactorsStr(), unexpectedLayouts, 0, nFactors)) { return; }
    size_t nRowsUsersModel = usersModel->getFactors()->getNumberOfRows();
    if(!checkNumericTable(usersModel->getIndices().get(), this->_errors.get(), usersIndicesStr(), unexpectedLayoutsIndices, 0, 1, nRowsUsersModel)) { return; }

    if(!checkNumericTable(itemsModel->getFactors().get(), this->_errors.get(), itemsFactorsStr(), unexpectedLayouts, 0, nFactors)) { return; }
    size_t nRowsItemsModel = itemsModel->getFactors()->getNumberOfRows();
    if(!checkNumericTable(itemsModel->getIndices().get(), this->_errors.get(), itemsIndicesStr(), unexpectedLayoutsIndices, 0, 1, nRowsItemsModel)) { return; }
}

}// namespace interface1
}// namespace ratings
}// namespace prediction
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
