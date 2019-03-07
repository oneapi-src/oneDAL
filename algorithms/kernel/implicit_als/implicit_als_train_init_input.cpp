/* file: implicit_als_train_init_input.cpp */
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

#include "implicit_als_training_init_types.h"
#include "daal_strings.h"

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
namespace init
{
namespace interface1
{
Input::Input(size_t nElements) : daal::algorithms::Input(nElements) {}

/**
 * Returns an input object for the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable,
           data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the number of items, that is, the number of columns in the input data set
 * \return Number of items
 */
size_t Input::getNumberOfItems() const
{
    return get(data)->getNumberOfRows();
}

/**
 * Checks the input objects and parameters of the implicit ALS initialization algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    if(method == defaultDense)
    {
        const int unexpectedLayouts = (int)NumericTableIface::upperPackedTriangularMatrix |
            (int)NumericTableIface::lowerPackedTriangularMatrix |
            (int)NumericTableIface::upperPackedSymmetricMatrix |
            (int)NumericTableIface::lowerPackedSymmetricMatrix;
        return checkNumericTable(get(data).get(), dataStr(), unexpectedLayouts);
    }
    const int expectedLayout = (int)NumericTableIface::csrArray;
    return checkNumericTable(get(data).get(), dataStr(), 0, expectedLayout);
}

}// namespace interface1
}// namespace init
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
