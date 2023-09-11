/* file: svd_result.cpp */
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
//  Implementation of svd classes.
//--
*/

#include "algorithms/svd/svd_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_SVD_RESULT_ID);

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the result of the SVD algorithm
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the SVD algorithm
 * \param[in] id    Identifier of the result
 * \param[in] value Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks final results of the algorithm
 * \param[in] input  Pointer to input objects
 * \param[in] par    Pointer to parameters
 * \param[in] method Computation method
 */
Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    const Input * algInput = static_cast<const Input *>(input);
    Parameter * svdPar     = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    size_t nVectors        = algInput->get(data)->getNumberOfRows();
    size_t nFeatures       = algInput->get(data)->getNumberOfColumns();
    int unexpectedLayouts  = (int)packed_mask;

    Status s = checkNumericTable(get(singularValues).get(), singularValuesStr(), unexpectedLayouts, 0, nFeatures, 1);
    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        s |= checkNumericTable(get(rightSingularMatrix).get(), rightSingularMatrixStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
    }
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        s |= checkNumericTable(get(leftSingularMatrix).get(), leftSingularMatrixStr(), unexpectedLayouts, 0, nFeatures, nVectors);
    }
    return s;
}
/**
 * Checks the result parameter of the SVD algorithm
 * \param[in] pres    Partial result of the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
Status Result::check(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * par, int method) const
{
    const OnlinePartialResult * algPartRes = static_cast<const OnlinePartialResult *>(pres);
    Parameter * svdPar                     = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    int unexpectedLayouts                  = (int)packed_mask;
    size_t nVectors                        = algPartRes->getNumberOfRows();
    size_t nFeatures                       = algPartRes->getNumberOfColumns();

    Status s = checkNumericTable(get(singularValues).get(), singularValuesStr(), unexpectedLayouts, 0, nFeatures, 1);
    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        s |= checkNumericTable(get(rightSingularMatrix).get(), rightSingularMatrixStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
    }
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        s |= checkNumericTable(get(leftSingularMatrix).get(), leftSingularMatrixStr(), unexpectedLayouts, 0, nFeatures, nVectors);
    }
    return s;
}

} // namespace svd
} // namespace algorithms
} // namespace daal
