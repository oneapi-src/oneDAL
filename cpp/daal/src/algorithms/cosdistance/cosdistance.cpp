/* file: cosdistance.cpp */
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
//  Implementation of cosine distance algorithm and types methods.
//--
*/

#include "algorithms/distance/cosine_distance_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

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
void Input::set(InputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the parameters of the cosine distance algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  computation method
*/
services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
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
void Result::set(ResultId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of the cosine distance algorithm
* \param[in] input   %Input of the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method
*/
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    const Input * algInput = static_cast<const Input *>(input);

    size_t nVectors       = algInput->get(data)->getNumberOfRows();
    int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray | (int)data_management::NumericTableIface::upperPackedTriangularMatrix
                            | (int)data_management::NumericTableIface::lowerPackedTriangularMatrix;

    return data_management::checkNumericTable(get(cosineDistance).get(), cosineDistanceStr(), unexpectedLayouts, 0, nVectors, nVectors);
}

} // namespace interface1
} // namespace cosine_distance
} // namespace algorithms
} // namespace daal
