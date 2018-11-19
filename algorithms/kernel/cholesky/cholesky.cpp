/* file: cholesky.cpp */
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
//  Implementation of cholesky algorithm and types methods.
//--
*/

#include "cholesky_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace cholesky
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_CHOLESKY_RESULT_ID);

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}

/**
 * Returns input NumericTable of the Cholesky algorithm
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input numeric table that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets input for the Cholesky algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks parameters of the Cholesky algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    NumericTablePtr inTable = get(data);

    DAAL_CHECK(inTable.get(), ErrorNullInputNumericTable);
    DAAL_CHECK(inTable->getNumberOfRows() , ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(inTable->getNumberOfColumns(), ErrorIncorrectNumberOfFeatures);

    NumericTableIface::StorageLayout iLayout = inTable->getDataLayout();

    DAAL_CHECK(inTable->getNumberOfColumns() == inTable->getNumberOfRows(), ErrorIncorrectSizeOfInputNumericTable);

    int iLayoutInt = (int) iLayout;
    if(iLayoutInt & data_management::packed_mask)
    {
        DAAL_CHECK(!(iLayout == NumericTableIface::lowerPackedTriangularMatrix ||
            iLayout == NumericTableIface::upperPackedTriangularMatrix), ErrorIncorrectTypeOfInputNumericTable);
    }
    return Status();
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

 /**
 * Returns result of the Cholesky algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the Cholesky algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the Cholesky algorithm
 * \param[in] input   %Input of algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    NumericTablePtr resTable = get(choleskyFactor);

    DAAL_CHECK(resTable.get(), ErrorNullInputNumericTable);
    DAAL_CHECK(resTable->getNumberOfRows() != 0, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(resTable->getNumberOfColumns() != 0, ErrorIncorrectNumberOfFeatures);

    NumericTableIface::StorageLayout rLayout = resTable->getDataLayout();

    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

    DAAL_CHECK((resTable->getNumberOfColumns() == algInput->get(data)->getNumberOfColumns()) &&
       (resTable->getNumberOfColumns() == resTable->getNumberOfRows()), ErrorIncorrectSizeOfOutputNumericTable);

    const int rLayoutInt = (int) rLayout;
    if(rLayoutInt & data_management::packed_mask)
    {
        DAAL_CHECK(rLayout == NumericTableIface::lowerPackedTriangularMatrix, ErrorIncorrectTypeOfOutputNumericTable);
    }
    return Status();
}


}// namespace interface1
}// namespace cholesky
}// namespace algorithms
}// namespace daal
