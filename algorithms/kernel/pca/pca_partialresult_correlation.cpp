/* file: pca_partialresult_correlation.cpp */
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "algorithms/pca/pca_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS3(PartialResult,correlationDense,SERIALIZATION_PCA_PARTIAL_RESULT_CORRELATION_ID);

PartialResult<correlationDense>::PartialResult() : PartialResultBase(lastPartialCorrelationResultId + 1) {};

/**
 * Gets partial results of the PCA Correlation algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
NumericTablePtr PartialResult<correlationDense>::get(PartialCorrelationResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

size_t PartialResult<correlationDense>::getNFeatures() const { return get(sumCorrelation)->getNumberOfColumns(); }

/**
 * Sets partial result of the PCA Correlation algorithm
 * \param[in] id      Identifier of the result
 * \param[in] value   Pointer to the object
 */
void PartialResult<correlationDense>::set(const PartialCorrelationResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
* Checks partial results of the PCA Correlation algorithm
* \param[in] input      %Input object of the algorithm
* \param[in] parameter  Algorithm %parameter
* \param[in] method     Computation method
*/
Status PartialResult<correlationDense>::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const InputIface *in = static_cast<const InputIface *>(input);
    DAAL_CHECK(!in->isCorrelation(), ErrorInputCorrelationNotSupportedInOnlineAndDistributed);
    return checkImpl(in->getNFeatures());
}

/**
* Checks partial results of the PCA Ccorrelation algorithm
* \param[in] par        Algorithm %parameter
* \param[in] method     Computation method
*/
Status PartialResult<correlationDense>::check(const daal::algorithms::Parameter *par, int method) const
{
    return checkImpl(0);
}

Status PartialResult<correlationDense>::checkImpl(size_t nFeatures) const
{
    int csrLayout = (int)NumericTableIface::csrArray;
    int packedLayouts = packed_mask;
    NumericTablePtr sumCorrelation = get(pca::sumCorrelation);
    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(pca::nObservationsCorrelation).get(), nObservationsCorrelationStr(), csrLayout, 0, 1, 1));
    DAAL_CHECK_STATUS(s, checkNumericTable(sumCorrelation.get(), sumCorrelationStr(), packedLayouts, 0, nFeatures, 1));
    nFeatures = sumCorrelation->getNumberOfColumns();
    return checkNumericTable(get(pca::crossProductCorrelation).get(), crossProductCorrelationStr(), packedLayouts, 0, nFeatures, nFeatures);
}

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
