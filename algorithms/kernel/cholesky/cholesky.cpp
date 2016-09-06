/* file: cholesky.cpp */
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
//  Implementation of cholesky algorithm and types methods.
//--
*/

#include "cholesky_types.h"

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

Input::Input() : daal::algorithms::Input(1) {}

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
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    if(Argument::size() != 1) { this->_errors->add(ErrorIncorrectNumberOfInputNumericTables); return; }

    NumericTablePtr inTable = get(data);

    if(inTable.get() == 0)                 { this->_errors->add(ErrorNullInputNumericTable); return; }
    if(inTable->getNumberOfRows() == 0)    { this->_errors->add(ErrorIncorrectNumberOfObservations); return; }
    if(inTable->getNumberOfColumns() == 0) { this->_errors->add(ErrorIncorrectNumberOfFeatures); return; }

    NumericTableIface::StorageLayout iLayout = inTable->getDataLayout();

    if(inTable->getNumberOfColumns() != inTable->getNumberOfRows())
    { this->_errors->add(ErrorIncorrectSizeOfInputNumericTable); return; }

    int iLayoutInt = (int) iLayout;
    if(iLayoutInt & data_management::packed_mask)
    {
        if(iLayout == NumericTableIface::lowerPackedTriangularMatrix ||
           iLayout == NumericTableIface::upperPackedTriangularMatrix)
        { this->_errors->add(ErrorIncorrectTypeOfInputNumericTable); return; }
    }
}


Result::Result() : daal::algorithms::Result(1) {}

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
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    if(Argument::size() != 1) { this->_errors->add(ErrorIncorrectNumberOfOutputNumericTables); return; }

    NumericTablePtr resTable = get(choleskyFactor);

    if(resTable.get() == 0)                 { this->_errors->add(ErrorNullInputNumericTable); return; }
    if(resTable->getNumberOfRows() == 0)    { this->_errors->add(ErrorIncorrectNumberOfObservations); return; }
    if(resTable->getNumberOfColumns() == 0) { this->_errors->add(ErrorIncorrectNumberOfFeatures); return; }

    NumericTableIface::StorageLayout rLayout = resTable->getDataLayout();

    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

    if(resTable->getNumberOfColumns() != algInput->get(data)->getNumberOfColumns() ||
       resTable->getNumberOfColumns() != resTable->getNumberOfRows())
    { this->_errors->add(ErrorIncorrectSizeOfOutputNumericTable); return; }

    int rLayoutInt = (int) rLayout;
    if(rLayoutInt & data_management::packed_mask)
    {
        if(rLayout != NumericTableIface::lowerPackedTriangularMatrix)
        { this->_errors->add(ErrorIncorrectTypeOfOutputNumericTable); return; }
    }
}


}// namespace interface1
}// namespace cholesky
}// namespace algorithms
}// namespace daal
