/* file: covariance_partialresult.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "covariance_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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

__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_COVARIANCE_PARTIAL_RESULT_ID);
PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1)
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
    return 0;
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
services::Status PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const InputIface *algInput = static_cast<const InputIface *>(input);
    size_t nFeatures = algInput->getNumberOfFeatures();
    return checkImpl(nFeatures);
}

/**
 * Check the correctness of PartialResult object
 * \param[in] parameter Pointer to the structure of the parameters of the algorithm
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Parameter *parameter, int method) const
{
    size_t nFeatures = getNumberOfFeatures();
    return checkImpl(nFeatures);
}

services::Status PartialResult::checkImpl(size_t nFeatures) const
{
    int unexpectedLayouts;
    services::Status s;

    unexpectedLayouts = (int)NumericTableIface::csrArray;
    s |= checkNumericTable(get(nObservations).get(),  nObservationsStr(), unexpectedLayouts, 0, 1, 1);
    if(!s) return s;

    unexpectedLayouts |= (int)NumericTableIface::upperPackedTriangularMatrix |
                         (int)NumericTableIface::lowerPackedTriangularMatrix;
    s |= checkNumericTable(get(crossProduct).get(), crossProductCorrelationStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
    if(!s) return s;

    unexpectedLayouts |= (int)NumericTableIface::upperPackedSymmetricMatrix |
                         (int)NumericTableIface::lowerPackedSymmetricMatrix;
    s |= checkNumericTable(get(sum).get(), sumStr(), unexpectedLayouts, 0, nFeatures, 1);
    return s;
}

}//namespace interface1

}//namespace covariance
}// namespace algorithms
}// namespace daal
