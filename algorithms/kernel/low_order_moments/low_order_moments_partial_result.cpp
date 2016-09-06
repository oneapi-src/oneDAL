/* file: low_order_moments_partial_result.cpp */
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
//  Implementation of LowOrderMoments classes.
//--
*/

#include "algorithms/moments/low_order_moments_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace interface1
{

PartialResult::PartialResult() : daal::algorithms::PartialResult(nPartialResults) {}

/**
 * Gets the number of columns in the partial result of the low order %moments algorithm
 * \return Number of columns in the partial result
 */
size_t PartialResult::getNumberOfColumns() const
{
    NumericTablePtr ntPtr = NumericTable::cast(Argument::get(partialMinimum));

    if(checkNumericTable(ntPtr.get(), this->_errors.get(), partialMinimumStr()))
    {
        return ntPtr->getNumberOfColumns();
    }
    else
    {
        return 0;
    }
}

/**
 * Returns the partial result of the low order %moments algorithm
 * \param[in] id   Identifier of the partial result, \ref PartialResultId
 * \return Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the partial result of the low order %moments algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks correctness of the partial result
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method
 */
void PartialResult::check(const daal::algorithms::Parameter *parameter, int method) const
{
    int unexpectedLayouts = (int)NumericTableIface::csrArray;
    if(!checkNumericTable(get(nObservations).get(), this->_errors.get(), nObservationsStr(), unexpectedLayouts, 0, 1, 1)) { return; }

    unexpectedLayouts = (int)packed_mask;
    if(!checkNumericTable(get(partialMinimum).get(), this->_errors.get(), partialMinimumStr(), unexpectedLayouts)) { return; }

    size_t nFeatures = get(partialMinimum)->getNumberOfColumns();
    checkImpl(nFeatures);
}

/**
 * Checks  the correctness of partial result
 * \param[in] input     Pointer to the structure with input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
void PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    size_t nFeatures = (static_cast<const InputIface *>(input))->getNumberOfColumns();

    int unexpectedLayouts = (int)NumericTableIface::csrArray;
    if(!checkNumericTable(get(nObservations).get(), this->_errors.get(), nObservationsStr(), unexpectedLayouts, 0, 1, 1)) { return; }

    checkImpl(nFeatures);
}

void PartialResult::checkImpl(size_t nFeatures) const
{
    int unexpectedLayouts = (int)packed_mask;
    const char* errorMessages[] = {partialMinimumStr(), partialMaximumStr(), partialSumStr(),
        partialSumSquaresStr(), partialSumSquaresCenteredStr() };

    for(size_t i = 1; i < nPartialResults; i++)
    {
        if(!checkNumericTable(get((PartialResultId)i).get(), this->_errors.get(), errorMessages[i-1],
            unexpectedLayouts, 0, nFeatures, 1)) { return; }
    }
}

void DefaultPartialResultInit::operator()(const Input &input, services::SharedPtr<PartialResult> &pres)
{
    /* Initializes number of rows with zero */
    data_management::NumericTablePtr nRowsTable = pres->get(nObservations);
    data_management::BlockDescriptor<int> nRowsBlock;
    int *nRows;
    nRowsTable->getBlockOfRows(0, 1, data_management::writeOnly, nRowsBlock);
    nRows = nRowsBlock.getBlockPtr();
    nRows[0] = 0;
    nRowsTable->releaseBlockOfRows(nRowsBlock);

    /* Gets first row of the input table */
    data_management::NumericTablePtr inTable = input.get(data);
    data_management::BlockDescriptor<double> firstRowBlock;

    double *firstRow;
    inTable->getBlockOfRows(0, 1, data_management::readOnly, firstRowBlock);
    firstRow = firstRowBlock.getBlockPtr();
    size_t nColumns = input.getNumberOfColumns();

    for(size_t i = 1; i < nPartialResults; i++)
    {
        data_management::NumericTablePtr nt = pres->get((PartialResultId)i);
        data_management::BlockDescriptor<double> partialEstimateBlock;
        double *partialEstimate;
        nt->getBlockOfRows(0, 1, data_management::writeOnly, partialEstimateBlock);
        partialEstimate = partialEstimateBlock.getBlockPtr();
        if(i == (size_t)partialMinimum || i == (size_t)partialMaximum)
        {
            /* Initializes partialMinimum and partialMaximum with the first row if the input table */
            for(size_t j = 0; j < nColumns; j++)
            {
                partialEstimate[j] = firstRow[j];
            }
        }
        else
        {
            /* Initializes the rest of partial estimates with zeros */
            for(size_t j = 0; j < nColumns; j++)
            {
                partialEstimate[j] = 0;
            }
        }
        nt->releaseBlockOfRows(partialEstimateBlock);
    }
    inTable->releaseBlockOfRows(firstRowBlock);
}

Parameter::Parameter() : initializationProcedure(new DefaultPartialResultInit()) {}

void Parameter::check() const
{
    if(!initializationProcedure) { this->_errors->add(services::ErrorIncorrectParameter); return; }
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
