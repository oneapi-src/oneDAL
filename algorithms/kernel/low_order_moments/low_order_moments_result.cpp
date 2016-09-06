/* file: low_order_moments_result.cpp */
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

Result::Result() : daal::algorithms::Result(nResults) {}

/**
 * Returns final result of the low order %moments algorithm
 * \param[in] id   identifier of the result, \ref ResultId
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets final result of the low order %moments algorithm
 * \param[in] id    Identifier of the final result
 * \param[in] value Pointer to the final result
 */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of result
 * \param[in] partialResult Pointer to the partial results
 * \param[in] par           %Parameter of the algorithm
 * \param[in] method        Computation method
 */
void Result::check(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *par, int method) const
{
    size_t nFeatures = static_cast<const PartialResult *>(partialResult)->get(partialMinimum)->getNumberOfColumns();
    checkImpl(nFeatures);
}

/**
 * Checks the correctness of result
 * \param[in] input     Pointer to the structure with input objects
 * \param[in] par       Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    size_t nFeatures = (static_cast<const InputIface *>(input))->getNumberOfColumns();
    checkImpl(nFeatures);
}

void Result::checkImpl(size_t nFeatures) const
{
    int unexpectedLayouts = (int)packed_mask;

    const char* errorMessages[] = {minimumStr(), maximumStr(), sumStr(), sumSquaresStr(), sumSquaresCenteredStr(), meanStr(),
        secondOrderRawMomentStr(), varianceStr(), standardDeviationStr(), variationStr() };

    for(size_t i = 0; i < nResults; i++)
    {
        if(!checkNumericTable(get((ResultId)i).get(), this->_errors.get(), errorMessages[i],
            unexpectedLayouts, 0, nFeatures, 1)) { return; }
    }
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
