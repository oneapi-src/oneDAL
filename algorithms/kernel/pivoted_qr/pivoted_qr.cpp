/* file: pivoted_qr.cpp */
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
//  Definition of Pivoted QR common types.
//--
*/

#include "pivoted_qr_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pivoted_qr
{
namespace interface1
{

Parameter::Parameter(const NumericTablePtr permutedColumns) : daal::algorithms::Parameter(), permutedColumns(permutedColumns) {}


Input::Input() : daal::algorithms::Input(1) {}

/**
 * Returns input object for the pivoted QR algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}
/**
 * Sets input object for the pivoted QR algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] value Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    if (!checkNumericTable(get(data).get(), this->_errors.get(), dataStr())) { return; }
    size_t nVectors = get(data)->getNumberOfRows();
    size_t nFeatures = get(data)->getNumberOfColumns();

    DAAL_CHECK_EX(nVectors >= nFeatures, ErrorIncorrectNumberOfRows, ArgumentName, dataStr());

    Parameter *parameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    if(parameter->permutedColumns.get() != NULL)
    {
        int unexpectedLayouts = (int)NumericTableIface::csrArray |
                                (int)NumericTableIface::upperPackedTriangularMatrix |
                                (int)NumericTableIface::lowerPackedTriangularMatrix |
                                (int)NumericTableIface::upperPackedSymmetricMatrix |
                                (int)NumericTableIface::lowerPackedSymmetricMatrix;

        if (!checkNumericTable(parameter->permutedColumns.get(), this->_errors.get(), permutedColumnsStr(),
            unexpectedLayouts, 0, nFeatures, 1)) { return; }
    }
}


Result::Result() : daal::algorithms::Result(3) {}

/**
 * Returns result of the pivoted QR algorithm
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets NumericTable to store the result of the pivoted QR algorithm
 * \param[in] id    Identifier of the result
 * \param[in] value Pointer to the storage NumericTable
 */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
* Checks the correctness of the result object
* \param[in] in     Pointer to the input objects structure
* \param[in] par    Pointer to the structure of the algorithm parameters
* \param[in] method Computation method
*/
void Result::check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const
{
    const Input *input = static_cast<const Input *>(in);

    size_t nVectors = input->get(data)->getNumberOfRows();
    size_t nFeatures = input->get(data)->getNumberOfColumns();

    int unexpectedLayouts = (int)NumericTableIface::csrArray |
                            (int)NumericTableIface::upperPackedTriangularMatrix |
                            (int)NumericTableIface::lowerPackedTriangularMatrix |
                            (int)NumericTableIface::upperPackedSymmetricMatrix |
                            (int)NumericTableIface::lowerPackedSymmetricMatrix;

    if (!checkNumericTable(get(matrixQ).get(), this->_errors.get(), matrixQStr(),
        unexpectedLayouts, 0, nFeatures, nVectors)) { return; }

    if (!checkNumericTable(get(permutationMatrix).get(), this->_errors.get(), permutationMatrixStr(),
        unexpectedLayouts, 0, nFeatures, 1)) { return; }

    unexpectedLayouts = (int)NumericTableIface::csrArray |
                        (int)NumericTableIface::lowerPackedTriangularMatrix |
                        (int)NumericTableIface::upperPackedSymmetricMatrix |
                        (int)NumericTableIface::lowerPackedSymmetricMatrix;

    if (!checkNumericTable(get(matrixR).get(), this->_errors.get(), matrixRStr(),
        unexpectedLayouts, 0, nFeatures, nFeatures)) { return; }
}

}// namespace interface1
}// namespace pivoted_qr
}// namespace algorithms
}// namespace daal
