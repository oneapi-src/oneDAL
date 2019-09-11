/* file: implicit_als_predict_ratings_partial_result.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "implicit_als_predict_ratings_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID);

/** Default constructor */
PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {}

/**
 * Returns a partial result of the rating prediction stage of the implicit ALS algorithm
 * \param[in] id    Identifier of the partial result
 * \return          Value that corresponds to the given identifier
 */
ResultPtr PartialResult::get(PartialResultId id) const
{
    return services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of the rating prediction stage of the implicit ALS algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the new partial result object
 */
void PartialResult::set(PartialResultId id, const ResultPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks a partial result of the implicit ALS algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    ResultPtr result = get(finalResult);
    const InputIface *algInput = static_cast<const InputIface *>(input);

    const size_t nUsers = algInput->getNumberOfUsers();
    const size_t nItems = algInput->getNumberOfItems();

    const int unexpectedLayouts = (int)packed_mask;
    if(result)
        return checkNumericTable(result->get(prediction).get(), predictionStr(), unexpectedLayouts, 0, nItems, nUsers);
    return services::Status();
}
}// namespace interface1
}// namespace ratings
}// namespace prediction
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
