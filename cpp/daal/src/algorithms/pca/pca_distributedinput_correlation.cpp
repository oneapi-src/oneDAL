/* file: pca_distributedinput_correlation.cpp */
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
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pca
{
DistributedInput<correlationDense>::DistributedInput() : InputIface(lastStep2MasterInputId + 1)
{
    Argument::set(partialResults, DataCollectionPtr(new DataCollection()));
}

DistributedInput<correlationDense>::DistributedInput(const DistributedInput<correlationDense> & other) : InputIface(other) {}

/**
 * Sets input objects for the PCA on the second step in the distributed processing mode
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Input object that corresponds to the given identifier
 */
void DistributedInput<correlationDense>::set(Step2MasterInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Gets input objects for the PCA on the second step in the distributed processing mode
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<correlationDense>::get(Step2MasterInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Retrieves specific partial result from the input objects of the PCA algorithm on the second step in the distributed processing mode
 * \param[in] id      Identifier of the partial result
 */
SharedPtr<PartialResult<correlationDense> > DistributedInput<correlationDense>::getPartialResult(size_t id) const
{
    DataCollectionPtr partialResultsCollection = staticPointerCast<DataCollection, SerializationIface>(get(partialResults));
    if (partialResultsCollection->size() <= id)
    {
        return SharedPtr<PartialResult<correlationDense> >();
    }
    return staticPointerCast<PartialResult<correlationDense>, SerializationIface>((*partialResultsCollection)[id]);
}

/**
 * Adds input objects of the PCA algorithm on the second step in the distributed processing mode
 * \param[in] id      Identifier of the argument
 * \param[in] value   Pointer to the argument
 */
void DistributedInput<correlationDense>::add(Step2MasterInputId id, const SharedPtr<PartialResult<correlationDense> > & value)
{
    DataCollectionPtr collection = staticPointerCast<DataCollection, SerializationIface>(get(id));
    collection->push_back(value);
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t DistributedInput<correlationDense>::getNFeatures() const
{
    return getPartialResult(0)->get(pca::crossProductCorrelation)->getNumberOfColumns();
}

/**
* Checks the input of the PCA algorithm
* \param[in] parameter Algorithm %parameter
* \param[in] method    Computation  method
*/
Status DistributedInput<correlationDense>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DataCollectionPtr collection = DataCollection::cast(Argument::get(partialResults));
    DAAL_CHECK(collection, ErrorNullPartialResultDataCollection);
    size_t nBlocks = collection->size();
    DAAL_CHECK(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables);

    for (size_t i = 0; i < nBlocks; i++)
    {
        SharedPtr<PartialResult<defaultDense> > partRes = staticPointerCast<PartialResult<defaultDense>, SerializationIface>((*collection)[i]);
        DAAL_CHECK(partRes, ErrorIncorrectElementInPartialResultCollection);
    }

    Status s;
    int packedLayouts = packed_mask;
    int csrLayout     = (int)NumericTableIface::csrArray;
    for (size_t j = 0; j < nBlocks; j++)
    {
        NumericTablePtr nObservationsCorrelation = getPartialResult(j)->get(pca::nObservationsCorrelation);
        DAAL_CHECK_STATUS(s, checkNumericTable(nObservationsCorrelation.get(), nObservationsCorrelationStr(), csrLayout, 0, 1, 1));

        NumericTablePtr crossProductCorrelation = getPartialResult(j)->get(pca::crossProductCorrelation);
        DAAL_CHECK_STATUS(s, checkNumericTable(crossProductCorrelation.get(), crossProductCorrelationStr(), packedLayouts));

        size_t nFeatures = getPartialResult(0)->get(pca::crossProductCorrelation)->getNumberOfColumns();
        DAAL_CHECK_STATUS(s, checkNumericTable(crossProductCorrelation.get(), crossProductCorrelationStr(), packedLayouts, 0, nFeatures, nFeatures));

        NumericTablePtr sumCorrelation = getPartialResult(j)->get(pca::sumCorrelation);
        DAAL_CHECK_STATUS(s, checkNumericTable(sumCorrelation.get(), sumCorrelationStr(), packedLayouts, 0, nFeatures, 1));
    }
    return s;
}

} // namespace pca
} // namespace algorithms
} // namespace daal
