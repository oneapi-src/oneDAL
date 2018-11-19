/* file: pca_input.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "algorithms/pca/pca_types.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{

InputIface::InputIface(size_t nElements) : daal::algorithms::Input(nElements), _isCorrelation(false) {};
InputIface::InputIface(const InputIface& other) : daal::algorithms::Input(other), _isCorrelation(other._isCorrelation) {};

Input::Input() : InputIface(lastInputDatasetId + 1) {};
Input::Input(const Input& other) : InputIface(other){}

/**
* Returns the input object of the PCA algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
*/
NumericTablePtr Input::get(InputDatasetId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets input dataset for the PCA algorithm
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the input object
 */
void Input::set(InputDatasetId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
    _isCorrelation = false;
}

/**
 * Sets input correlation matrix for the PCA algorithm
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the input object
 */
void Input::set(InputCorrelationId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
    _isCorrelation = true;
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t Input::getNFeatures() const
{
    return (staticPointerCast<NumericTable, SerializationIface>(Argument::get(data)))->getNumberOfColumns();
}

/**
* Checks input algorithm parameters
* \param[in] par     Algorithm %parameter
* \param[in] method  Computation method
* \return Errors detected while checking
*/
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfInputNumericTables);
    NumericTablePtr dataTable = get(data);
    Status s;
    if(_isCorrelation)
    {
        int unexpectedLayouts = (int)NumericTableIface::upperPackedTriangularMatrix | (int)NumericTableIface::lowerPackedTriangularMatrix;
        DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), correlationStr(), unexpectedLayouts));
        DAAL_CHECK_EX(dataTable->getNumberOfColumns() == dataTable->getNumberOfRows(), ErrorNumericTableIsNotSquare, ArgumentName, correlationStr());
    }
    else
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));
        if (method == svdDense)
        {
            DAAL_CHECK_EX(dataTable->getNumberOfColumns() <= dataTable->getNumberOfRows(), ErrorIncorrectNumberOfRows, ArgumentName, dataStr());
        }
    }
    return s;
}

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
