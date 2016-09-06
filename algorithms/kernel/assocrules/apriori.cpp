/* file: apriori.cpp */
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
//  Implementation of the interface for the association rules algorithm
//--
*/

#include "apriori_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace interface1
{

Parameter::Parameter(double minSupport, double minConfidence, size_t nUniqueItems, size_t nTransactions, bool discoverRules,
                        ItemsetsOrder itemsetsOrder, RulesOrder rulesOrder, size_t minSize, size_t maxSize) :
        minSupport(minSupport),
        minConfidence(minConfidence),
        nUniqueItems(nUniqueItems),
        nTransactions(nTransactions),
        discoverRules(discoverRules),
        itemsetsOrder(itemsetsOrder),
        rulesOrder(rulesOrder),
        minItemsetSize(minSize),
        maxItemsetSize(maxSize)
    {}

/**
 * Checks parameters of the association rules algorithm
 */
void Parameter::check() const
{
    DAAL_CHECK_EX(minSupport >= 0 && minSupport < 1, ErrorIncorrectParameter, ParameterName, minSupportStr());
    DAAL_CHECK_EX(minConfidence >= 0 && minConfidence < 1, ErrorIncorrectParameter, ParameterName, minConfidenceStr());
    DAAL_CHECK_EX(minItemsetSize <= maxItemsetSize, ErrorIncorrectParameter, ParameterName, minItemsetSizeStr());
}

Input::Input() : daal::algorithms::Input(1) {}

/**
 * Returns the input object of the association rules algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the input object of the association rules algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks parameters of the association rules algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method of the algorithm
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfInputNumericTables);

    int unexpectedLayouts = (int)NumericTableIface::upperPackedSymmetricMatrix  |
                            (int)NumericTableIface::lowerPackedSymmetricMatrix  |
                            (int)NumericTableIface::upperPackedTriangularMatrix |
                            (int)NumericTableIface::lowerPackedTriangularMatrix;
    size_t nColumns = 2;
    if (!checkNumericTable(get(data).get(), this->_errors.get(), dataStr(), unexpectedLayouts, 0, nColumns)) { return; }

    size_t nRows = get(data)->getNumberOfRows();
    Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

    DAAL_CHECK_EX(algParameter->nUniqueItems <= nRows, ErrorIncorrectParameter, ParameterName, nUniqueItemsStr());
    DAAL_CHECK_EX(algParameter->nTransactions <= nRows, ErrorIncorrectParameter, ParameterName, nTransactionsStr());
    DAAL_CHECK_EX((algParameter->maxItemsetSize <= nRows) && (algParameter->nUniqueItems <= 0 || algParameter->maxItemsetSize <= algParameter->nUniqueItems),
        ErrorIncorrectParameter, ParameterName, maxItemsetSizeStr());
}


Result::Result() : daal::algorithms::Result(5) {}

/**
 * Returns the final result of the association rules algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the final result of the association rules algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the association rules algorithm
 * \param[in] input   %Input of algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    checkNT(get(largeItemsets), this->_errors, largeItemsetsStr(), 2);
    checkNT(get(largeItemsetsSupport), this->_errors, largeItemsetsSupportStr(), 2);

    if(algParameter->discoverRules)
    {
        DAAL_CHECK(Argument::size() == 5, ErrorIncorrectNumberOfOutputNumericTables);
        checkNT(get(antecedentItemsets), this->_errors, antecedentItemsetsStr(), 2);
        checkNT(get(consequentItemsets), this->_errors, consequentItemsetsStr(), 2);
        checkNT(get(confidence), this->_errors, confidenceStr(), 1);
    }
    else
    {
        DAAL_CHECK(Argument::size() == 2, ErrorIncorrectNumberOfOutputNumericTables);
    }
}

void Result::checkNT(NumericTablePtr nt, SharedPtr<ErrorCollection> errors, const char *description, size_t nColumns) const
{
    int unexpectedLayouts = packed_mask;
    DAAL_CHECK_EX(nt, ErrorNullNumericTable, ArgumentName, description);

    if(nt->getDataMemoryStatus() == NumericTableIface::notAllocated)
    {
        DAAL_CHECK_EX(nt->getNumberOfColumns() == nColumns, ErrorIncorrectNumberOfColumns, ArgumentName, description);
        DAAL_CHECK_EX(((int)nt->getDataLayout() & unexpectedLayouts) == 0, ErrorIncorrectTypeOfNumericTable, ArgumentName, description);
    }
    else
    {
        if (!checkNumericTable(nt.get(), errors.get(), description, unexpectedLayouts, 0, nColumns)) { return; }
    }
}

}// namespace interface1
}// namespace association_rules
}// namespace algorithms
}// namespace daal
