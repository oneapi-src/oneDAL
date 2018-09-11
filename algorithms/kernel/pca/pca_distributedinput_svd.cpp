/* file: pca_distributedinput_svd.cpp */
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

DistributedInput<svdDense>::DistributedInput() : InputIface(lastStep2MasterInputId + 1)
{
    Argument::set(partialResults, DataCollectionPtr(new DataCollection()));
}
DistributedInput<svdDense>::DistributedInput(const DistributedInput<svdDense>& other) : InputIface(other){}

/**
 * Sets input objects for the PCA on the second step in the distributed processing mode
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Input object that corresponds to the given identifier
 */
void DistributedInput<svdDense>::set(Step2MasterInputId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Gets input objects for the PCA algorithm on the second step in the distributed processing mode
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<svdDense>::get(Step2MasterInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Adds input objects of the PCA algorithm on the second step in the distributed processing mode
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the input object
 */
void DistributedInput<svdDense>::add(Step2MasterInputId id, const SharedPtr<PartialResult<svdDense> > &value)
{
    DataCollectionPtr collection = staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
    collection->push_back(value);
}

/**
 * Retrieves specific partial result from the input objects of the PCA algorithm on the second step in the distributed processing mode
 * \param[in] id      Identifier of the partial result
 */
SharedPtr<PartialResult<svdDense> > DistributedInput<svdDense>::getPartialResult(size_t id) const
{
    DataCollectionPtr partialResultsCollection = staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));
    if(partialResultsCollection->size() <= id)
    {
        return SharedPtr<PartialResult<svdDense> >();
    }
    return staticPointerCast<PartialResult<svdDense>, SerializationIface>((*partialResultsCollection)[id]);
}

/**
* Checks the input of the PCA algorithm
* \param[in] parameter Algorithm %parameter
* \param[in] method    Computation  method
*/
Status DistributedInput<svdDense>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    DataCollectionPtr collection = DataCollection::cast(Argument::get(partialResults));
    DAAL_CHECK(collection, ErrorNullPartialResultDataCollection);
    size_t nBlocks = collection->size();
    DAAL_CHECK(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables);

    for(size_t i = 0; i < nBlocks; i++)
    {
        SharedPtr<PartialResult<svdDense> > partRes = staticPointerCast<PartialResult<svdDense>, SerializationIface>((*collection)[i]);
        DAAL_CHECK(partRes, ErrorIncorrectElementInPartialResultCollection);
    }

    int packedLayouts = packed_mask;
    int csrLayout = (int)NumericTableIface::csrArray;

    Status s;
    for(size_t j = 0; j < nBlocks; j++)
    {
        NumericTablePtr nObservationsSVD = getPartialResult(j)->get(pca::nObservationsSVD);
        DAAL_CHECK_STATUS(s, checkNumericTable(nObservationsSVD.get(), nObservationsSVDStr(), csrLayout, 0, 1, 1));

        NumericTablePtr sumSquaresSVD = getPartialResult(j)->get(pca::sumSquaresSVD);
        DAAL_CHECK_STATUS(s, checkNumericTable(sumSquaresSVD.get(), sumSquaresSVDStr(), packedLayouts, 0, 0, 1));

        size_t nFeatures = getPartialResult(0)->get(pca::sumSquaresSVD)->getNumberOfColumns();
        NumericTablePtr sumSVD = getPartialResult(j)->get(pca::sumSVD);

        DAAL_CHECK_STATUS(s, checkNumericTable(sumSVD.get(), sumSVDStr(), packedLayouts, 0, nFeatures, 1));
        DAAL_CHECK(sumSquaresSVD->getNumberOfColumns() == sumSVD->getNumberOfColumns(), ErrorIncorrectNumberOfColumns);

        DataCollectionPtr auxiliaryData = getPartialResult(j)->get(pca::auxiliaryData);
        DAAL_CHECK(auxiliaryData, ErrorNullAuxiliaryDataCollection);
        DAAL_CHECK(auxiliaryData->size() > 0, ErrorEmptyAuxiliaryDataCollection);

        for(size_t i = 0; i < auxiliaryData->size(); i++)
        {
            NumericTablePtr table = NumericTable::cast((*auxiliaryData)[i]);
            DAAL_CHECK(table, ErrorIncorrectElementInNumericTableCollection);
            DAAL_CHECK_STATUS(s, checkNumericTable(table.get(), auxiliaryDataStr(), csrLayout, 0, nFeatures, nFeatures));
        }
    }
    return s;
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t DistributedInput<svdDense>::getNFeatures() const
{
    return getPartialResult(0)->get(pca::sumSVD)->getNumberOfColumns();
}

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
