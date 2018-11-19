/* file: low_order_moments_result.cpp */
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
//  Implementation of LowOrderMoments classes.
//--
*/

#include "algorithms/moments/low_order_moments_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_MOMENTS_RESULT_ID);

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

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
services::Status Result::check(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *par, int method) const
{
    size_t nFeatures = static_cast<const PartialResult *>(partialResult)->get(partialMinimum)->getNumberOfColumns();
    return checkImpl(nFeatures);
}

/**
 * Checks the correctness of result
 * \param[in] input     Pointer to the structure with input objects
 * \param[in] par       Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    size_t nFeatures = 0;
    services::Status s;
    DAAL_CHECK_STATUS(s, (static_cast<const InputIface *>(input))->getNumberOfColumns(nFeatures));
    return checkImpl(nFeatures);
}

services::Status Result::checkImpl(size_t nFeatures) const
{
    services::Status s;
    const int unexpectedLayouts = (int)packed_mask;

    const char* errorMessages[] = {minimumStr(), maximumStr(), sumStr(), sumSquaresStr(), sumSquaresCenteredStr(), meanStr(),
        secondOrderRawMomentStr(), varianceStr(), standardDeviationStr(), variationStr() };

    for(size_t i = 0; i < lastResultId + 1; i++)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get((ResultId)i).get(), errorMessages[i],
            unexpectedLayouts, 0, nFeatures, 1));
    }
    return s;
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
