/* file: pca_distributedinput_correlation.cpp */
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "algorithms/pca/pca_types.h"
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

DistributedInput<correlationDense>::DistributedInput() : InputIface(lastStep2MasterInputId + 1)
{
    Argument::set(partialResults, DataCollectionPtr(new DataCollection()));
}

DistributedInput<correlationDense>::DistributedInput(const DistributedInput<correlationDense>& other) : InputIface(other){}

/**
 * Sets input objects for the PCA on the second step in the distributed processing mode
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Input object that corresponds to the given identifier
 */
void DistributedInput<correlationDense>::set(Step2MasterInputId id, const DataCollectionPtr &ptr)
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
    if(partialResultsCollection->size() <= id)
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
void DistributedInput<correlationDense>::add(Step2MasterInputId id, const SharedPtr<PartialResult<correlationDense> > &value)
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
Status DistributedInput<correlationDense>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    DataCollectionPtr collection = DataCollection::cast(Argument::get(partialResults));
    DAAL_CHECK(collection, ErrorNullPartialResultDataCollection);
    size_t nBlocks = collection->size();
    DAAL_CHECK(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables);

    for(size_t i = 0; i < nBlocks; i++)
    {
        SharedPtr<PartialResult<defaultDense> > partRes = staticPointerCast<PartialResult<defaultDense>, SerializationIface>((*collection)[i]);
        DAAL_CHECK(partRes, ErrorIncorrectElementInPartialResultCollection);
    }

    Status s;
    int packedLayouts = packed_mask;
    int csrLayout = (int)NumericTableIface::csrArray;
    for(size_t j = 0; j < nBlocks; j++)
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

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
