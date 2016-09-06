/* file: covariance_partialresult.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "covariance_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{

PartialResult::PartialResult() : daal::algorithms::PartialResult(3)
    {}

/**
 * Gets the number of columns in the partial result of the correlation or variance-covariance matrix algorithm
 * \return Number of columns in the partial result
 */
size_t PartialResult::getNumberOfFeatures() const
{
    NumericTablePtr ntPtr = NumericTable::cast(Argument::get(crossProduct));
    if(ntPtr)
    {
        return ntPtr->getNumberOfColumns();
    }
    else
    {
        this->_errors->add(ErrorIncorrectSizeOfInputNumericTable);
        return 0;
    }
}

/**
 * Returns the partial result of the correlation or variance-covariance matrix algorithm
 * \param[in] id   Identifier of the partial result, \ref PartialResultId
 * \return Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the partial result of the correlation or variance-covariance matrix algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Check correctness of the partial result
 * \param[in] input     Pointer to the structure with input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
void PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const InputIface *algInput = static_cast<const InputIface *>(input);
    size_t nFeatures = algInput->getNumberOfFeatures();
    checkImpl(nFeatures);
}

/**
 * Check the correctness of PartialResult object
 * \param[in] parameter Pointer to the structure of the parameters of the algorithm
 * \param[in] method    Computation method
 */
void PartialResult::check(const daal::algorithms::Parameter *parameter, int method) const
{
    size_t nFeatures = getNumberOfFeatures();
    checkImpl(nFeatures);
}

void PartialResult::checkImpl(size_t nFeatures) const
{
    int unexpectedLayouts;

    unexpectedLayouts = (int)NumericTableIface::csrArray;
    if (!checkNumericTable(get(nObservations).get(), this->_errors.get(),
                                            nObservationsStr(), unexpectedLayouts, 0, 1, 1)) { return; }

    unexpectedLayouts |= (int)NumericTableIface::upperPackedTriangularMatrix |
                         (int)NumericTableIface::lowerPackedTriangularMatrix;
    if (!checkNumericTable(get(crossProduct).get(), this->_errors.get(),
                                            crossProductCorrelationStr(), unexpectedLayouts, 0, nFeatures, nFeatures)) { return; }

    unexpectedLayouts |= (int)NumericTableIface::upperPackedSymmetricMatrix |
                         (int)NumericTableIface::lowerPackedSymmetricMatrix;
    if (!checkNumericTable(get(sum).get(), this->_errors.get(),
                                            sumStr(), unexpectedLayouts, 0, nFeatures, 1)) { return; }
}

void PartialResultsInitIface::setToZero(data_management::NumericTable *table)
{
    data_management::BlockDescriptor<double> block;
    size_t nCols = table->getNumberOfColumns();
    size_t nRows = table->getNumberOfRows();

    double *data;
    table->getBlockOfRows(0, nRows, data_management::writeOnly, block);
    data = block.getBlockPtr();

    for(size_t i = 0; i < nCols * nRows; i++)
    {
        data[i] = 0.0;
    };

    table->releaseBlockOfRows(block);
};

/**
 * Initializes partial results of the correlation or variance-covariance matrix algorithm
 * \param[in]       input     %Input objects of the algorithm
 * \param[in,out]   pres      Partial results of the algorithm
 */
void DefaultPartialResultsInit::operator()(const Input &input, services::SharedPtr<PartialResult> &pres)
{
    setToZero(pres->get(nObservations).get());
    setToZero(pres->get(crossProduct).get());
    setToZero(pres->get(sum).get());
}

}//namespace interface1

}//namespace covariance
}// namespace algorithms
}// namespace daal
