/* file: low_order_moments_distributed_input.cpp */
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

template<>
DistributedInput<step2Master>::DistributedInput() : InputIface(lastMasterInputId + 1)
{
    Argument::set(partialResults, DataCollectionPtr(new DataCollection()));
}

template<>
DistributedInput<step2Master>::DistributedInput(const DistributedInput<step2Master>& other) : InputIface(other){}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
template<>
services::Status DistributedInput<step2Master>::getNumberOfColumns(size_t& nCols) const
{
    DataCollectionPtr collectionOfPartialResults =
        staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));

    DAAL_CHECK(collectionOfPartialResults, ErrorNullInputDataCollection);
    DAAL_CHECK(collectionOfPartialResults->size(), ErrorIncorrectNumberOfInputNumericTables);

    PartialResultPtr partialResult = PartialResult::cast((*collectionOfPartialResults)[0]);

    DAAL_CHECK(partialResult.get(), ErrorIncorrectElementInPartialResultCollection);
    NumericTablePtr ntPtr = partialResult->get(partialMinimum);

    Status s = checkNumericTable(ntPtr.get(), partialMinimumStr());
    nCols = s ? ntPtr->getNumberOfColumns() : 0;
    return s;
}

/**
 * Adds partial result to the collection of input objects for the low order moments algorithm in the distributed processing mode.
 * \param[in] id            Identifier of the input object
 * \param[in] partialResult Partial result obtained in the first step of the distributed algorithm
 */
template<>
void DistributedInput<step2Master>::add(MasterInputId id, const PartialResultPtr &partialResult)
{
    DataCollectionPtr collection =
        staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
    collection->push_back(staticPointerCast<SerializationIface, PartialResult>(partialResult));
}
/**
 * Sets input object for the low order moments algorithm in the distributed processing mode.
 * \param[in] id  Identifier of the input object
 * \param[in] ptr Pointer to the input object
 */
template<>
void DistributedInput<step2Master>::set(MasterInputId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}
/**
 * Returns the collection of input objects
 * \param[in] id   Identifier of the input object, \ref MasterInputId
 * \return Collection of distributed input objects
 */
template<>
DataCollectionPtr DistributedInput<step2Master>::get(MasterInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));
}
/**
 * Checks algorithm parameters on the master node
 * \param[in] parameter Pointer to the algorithm parameters
 * \param[in] method    Computation method
 */
template<>
services::Status DistributedInput<step2Master>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DataCollectionPtr collectionPtr = DataCollection::cast(Argument::get(0));
    DAAL_CHECK(collectionPtr, ErrorNullInputDataCollection);
    size_t nBlocks = collectionPtr->size();
    DAAL_CHECK(collectionPtr->size() != 0, ErrorIncorrectNumberOfInputNumericTables);

    for(size_t j = 0; j < nBlocks; j++)
    {
        PartialResultPtr partialResult = PartialResult::cast((*collectionPtr)[j]);
        DAAL_CHECK(partialResult.get() != 0, ErrorIncorrectElementInPartialResultCollection);

        /* Checks partial number of observations */
        int unexpectedLayouts = (int)NumericTableIface::csrArray;
        DAAL_CHECK_STATUS(s, checkNumericTable(partialResult->get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, 1));

        unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS(s, checkNumericTable(partialResult->get(partialMinimum).get(), partialMinimumStr(), unexpectedLayouts));

        size_t nFeatures = partialResult->get(partialMinimum)->getNumberOfColumns();
        const char* errorMessages[] = {partialMinimumStr(), partialMaximumStr(), partialSumStr(), partialSumSquaresStr(), partialSumSquaresCenteredStr() };

        for(size_t i = 1; i < lastPartialResultId + 1; i++)
            DAAL_CHECK_STATUS(s, checkNumericTable(partialResult->get((PartialResultId)i).get(), errorMessages[i - 1], unexpectedLayouts, 0, nFeatures, 1));
        }
    return s;
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
