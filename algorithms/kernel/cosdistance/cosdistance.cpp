/* file: cosdistance.cpp */
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
//  Implementation of cosine distance algorithm and types methods.
//--
*/

#include "cosine_distance_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_COSINE_DISTANCE_RESULT_ID);
Input::Input() : daal::algorithms::Input(lastInputId + 1) {}

/**
* Returns the input object of the cosine distance algorithm
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
* Sets the input object for the cosine distance algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the parameters of the cosine distance algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  computation method
*/
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    return data_management::checkNumericTable(get(data).get(), dataStr());
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}


/**
 * Returns the result of the cosine distance algorithm
 * \param[in] id   Identifier of the result
 * \return         %Result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the cosine distance algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the result object
 */
void Result::set(ResultId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of the cosine distance algorithm
* \param[in] input   %Input of the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method
*/
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Input *algInput = static_cast<const Input *>(input);

    size_t nVectors  = algInput->get(data)->getNumberOfRows();
    int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray |
                            (int)data_management::NumericTableIface::upperPackedTriangularMatrix |
                            (int)data_management::NumericTableIface::lowerPackedTriangularMatrix;

    return data_management::checkNumericTable(get(cosineDistance).get(), cosineDistanceStr(), unexpectedLayouts, 0, nVectors, nVectors);
}

}// namespace interface1
}// namespace cosine_distance
}// namespace algorithms
}// namespace daal
