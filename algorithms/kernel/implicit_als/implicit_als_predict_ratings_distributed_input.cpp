/* file: implicit_als_predict_ratings_distributed_input.cpp */
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

DistributedInput<step1Local>::DistributedInput() : InputIface(2) {}

/**
 * Returns an input object for the rating prediction stage of the implicit ALS algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
PartialModelPtr DistributedInput<step1Local>::get(PartialModelInputId id) const
{
    return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the rating prediction stage of the implicit ALS algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the new input object value
 */
void DistributedInput<step1Local>::set(PartialModelInputId id, const PartialModelPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the number of rows in the input numeric table
 * \return Number of rows in the input numeric table
 */
size_t DistributedInput<step1Local>::getNumberOfUsers() const
{
    PartialModelPtr usersModel = get(usersPartialModel);
    if(usersModel)
    {
        data_management::NumericTablePtr factors = usersModel->getFactors();
        if(factors)
            return factors->getNumberOfRows();
    }
    return 0;
}

/**
 * Returns the number of columns in the input numeric table
 * \return Number of columns in the input numeric table
 */
size_t DistributedInput<step1Local>::getNumberOfItems() const
{
    PartialModelPtr itemsModel = get(itemsPartialModel);
    if(itemsModel)
    {
        data_management::NumericTablePtr factors = itemsModel->getFactors();
        if(factors)
            return factors->getNumberOfRows();
    }
    return 0;
}

/**
 * Checks the parameters of the rating prediction stage of the implicit ALS algorithm
 * \param[in] parameter     Algorithm %parameter
 * \param[in] method        Computation method for the algorithm
 */
services::Status DistributedInput<step1Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors = algParameter->nFactors;

    PartialModelPtr usersModel = get(usersPartialModel);
    PartialModelPtr itemsModel = get(itemsPartialModel);
    DAAL_CHECK(usersModel, ErrorNullPartialModel);
    DAAL_CHECK(itemsModel, ErrorNullPartialModel);

    const int unexpectedLayouts = (int)NumericTableIface::upperPackedTriangularMatrix |
        (int)NumericTableIface::lowerPackedTriangularMatrix |
        (int)NumericTableIface::upperPackedSymmetricMatrix |
        (int)NumericTableIface::lowerPackedSymmetricMatrix;

    const int unexpectedLayoutsIndices = unexpectedLayouts | (int)NumericTableIface::csrArray;
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(usersModel->getFactors().get(), usersFactorsStr(), unexpectedLayouts, 0, nFactors));
    const size_t nRowsUsersModel = usersModel->getFactors()->getNumberOfRows();
    DAAL_CHECK_STATUS(s, checkNumericTable(usersModel->getIndices().get(), usersIndicesStr(), unexpectedLayoutsIndices, 0, 1, nRowsUsersModel));

    DAAL_CHECK_STATUS(s, checkNumericTable(itemsModel->getFactors().get(), itemsFactorsStr(), unexpectedLayouts, 0, nFactors));
    const size_t nRowsItemsModel = itemsModel->getFactors()->getNumberOfRows();
    return checkNumericTable(itemsModel->getIndices().get(), itemsIndicesStr(), unexpectedLayoutsIndices, 0, 1, nRowsItemsModel);
}

}// namespace interface1
}// namespace ratings
}// namespace prediction
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
