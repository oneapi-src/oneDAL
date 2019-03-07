/* file: qr_result.cpp */
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
//  Implementation of qr classes.
//--
*/

#include "algorithms/qr/qr_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_QR_RESULT_ID);

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the result of the QR decomposition algorithm
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the QR decomposition algorithm
 * \param[in] id    Identifier of the result
 * \param[in] value Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks final results of the algorithm
 * \param[in] input  Pointer to input objects
 * \param[in] par    Pointer to parameters
 * \param[in] method Computation method
 */
Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    size_t nVectors = algInput->get(data)->getNumberOfRows();
    size_t nFeatures = algInput->get(data)->getNumberOfColumns();
    int unexpectedLayouts = (int)packed_mask;

    Status s = checkNumericTable(get(matrixQ).get(), matrixQStr(), unexpectedLayouts, 0, nFeatures, nVectors);
    if(!s) { return s; }

    s |= checkNumericTable(get(matrixR).get(), matrixRStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
    return s;
}

/**
 * Checks the result parameter of the QR algorithm
 * \param[in] pres    Partial result of the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
Status Result::check(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *par, int method) const
{
    const OnlinePartialResult  *algPartRes = static_cast<const OnlinePartialResult *>(pres);
    int unexpectedLayouts = (int)packed_mask;
    size_t nVectors = algPartRes->getNumberOfRows();
    size_t nFeatures = algPartRes->getNumberOfColumns();

    Status s = checkNumericTable(get(matrixQ).get(), matrixQStr(), unexpectedLayouts, 0, nFeatures, nVectors);
    if(!s) { return s; }

    s |= checkNumericTable(get(matrixR).get(), matrixRStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
    return s;
}

} // namespace interface1
} // namespace qr
} // namespace algorithm
} // namespace daal
