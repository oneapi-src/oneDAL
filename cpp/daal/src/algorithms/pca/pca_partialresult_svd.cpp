/* file: pca_partialresult_svd.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "algorithms/pca/pca_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{
__DAAL_REGISTER_SERIALIZATION_CLASS3(PartialResult, svdDense, SERIALIZATION_PCA_PARTIAL_RESULT_SVD_ID);

PartialResult<svdDense>::PartialResult() : PartialResultBase(lastPartialSVDCollectionResultId + 1) {};

/**
* Gets partial results of the PCA SVD algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
*/
NumericTablePtr PartialResult<svdDense>::get(PartialSVDTableResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

size_t PartialResult<svdDense>::getNFeatures() const
{
    return get(sumSVD)->getNumberOfColumns();
}

/**
* Gets partial results of the PCA SVD algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
*/
DataCollectionPtr PartialResult<svdDense>::get(PartialSVDCollectionResultId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
* Gets partial results of the PCA SVD algorithm
 * \param[in] id            Identifier of the input object
 * \param[in] elementId     Identifier of the collection element
 * \return                  Input object that corresponds to the given identifier
*/
NumericTablePtr PartialResult<svdDense>::get(PartialSVDCollectionResultId id, const size_t & elementId) const
{
    DataCollectionPtr collection = get(id);
    if (!collection.get() || elementId >= collection->size()) return NumericTablePtr();
    return staticPointerCast<NumericTable, SerializationIface>((*collection)[elementId]);
}

/**
 * Sets partial result of the PCA SVD algorithm
 * \param[in] id      Identifier of the result
 * \param[in] value   Pointer to  the object
 */
void PartialResult<svdDense>::set(PartialSVDTableResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Sets partial result of the PCA SVD algorithm
 * \param[in] id      Identifier of the result
 * \param[in] value   Pointer to the object
 */
void PartialResult<svdDense>::set(PartialSVDCollectionResultId id, const DataCollectionPtr & value)
{
    Argument::set(id, staticPointerCast<SerializationIface, DataCollection>(value));
}

/**
 * Adds partial result of the PCA SVD algorithm
 * \param[in] id      Identifier of the argument
 * \param[in] value   Pointer to the object
 */
void PartialResult<svdDense>::add(const PartialSVDCollectionResultId & id, const DataCollectionPtr & value)
{
    DataCollectionPtr collection = staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
    collection->push_back(value);
}

/**
* Checks partial results of the PCA SVD algorithm
* \param[in] input      %Input of algorithm
* \param[in] parameter  %Parameter of algorithm
* \param[in] method     Computation method
*/
Status PartialResult<svdDense>::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const InputIface * in = static_cast<const InputIface *>(input);
    DAAL_CHECK(!in->isCorrelation(), ErrorInputCorrelationNotSupportedInOnlineAndDistributed);
    return checkImpl(in->getNFeatures());
}

/**
* Checks partial results of the PCA SVD algorithm
* \param[in] method     Computation method
* \param[in] par        %Parameter of algorithm
*/
Status PartialResult<svdDense>::check(const daal::algorithms::Parameter * par, int method) const
{
    return checkImpl(0);
}

Status PartialResult<svdDense>::checkImpl(size_t nFeatures) const
{
    int packedLayouts             = packed_mask;
    int csrLayout                 = (int)NumericTableIface::csrArray;
    NumericTablePtr sumSquaresSVD = get(pca::sumSquaresSVD);

    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(pca::nObservationsSVD).get(), nObservationsSVDStr(), csrLayout, 0, 1, 1));
    DAAL_CHECK_STATUS(s, checkNumericTable(sumSquaresSVD.get(), sumSquaresSVDStr(), packedLayouts, 0, nFeatures, 1));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(pca::sumSVD).get(), sumSVDStr(), packedLayouts, 0, sumSquaresSVD->getNumberOfColumns(), 1));
    DAAL_CHECK(get(pca::auxiliaryData), ErrorNullAuxiliaryDataCollection);
    return s;
}

} // namespace pca
} // namespace algorithms
} // namespace daal
