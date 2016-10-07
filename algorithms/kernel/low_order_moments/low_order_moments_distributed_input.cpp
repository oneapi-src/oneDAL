/* file: low_order_moments_distributed_input.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of LowOrderMoments classes.
//--
*/

#include "algorithms/moments/low_order_moments_types.h"

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

DistributedInput<step2Master>::DistributedInput() : InputIface(1)
{
    Argument::set(partialResults, DataCollectionPtr(new DataCollection()));
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t DistributedInput<step2Master>::getNumberOfColumns() const
{
    DataCollectionPtr collectionOfPartialResults =
        staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));

    if(!collectionOfPartialResults) { this->_errors->add(ErrorNullInputDataCollection); return 0; }
    else if(collectionOfPartialResults->size() == 0) { this->_errors->add(ErrorIncorrectNumberOfInputNumericTables); return 0; }

    PartialResultPtr partialResult = PartialResult::cast((*collectionOfPartialResults)[0]);

    if(partialResult.get() == 0) { this->_errors->add(ErrorIncorrectElementInPartialResultCollection); return 0; }
    NumericTablePtr ntPtr = partialResult->get(partialMinimum);

    if(checkNumericTable(ntPtr.get(), this->_errors.get(), partialMinimumStr()))
    {
        return ntPtr->getNumberOfColumns();
    }
    return 0;
}
/**
 * Adds partial result to the collection of input objects for the low order moments algorithm in the distributed processing mode.
 * \param[in] id            Identifier of the input object
 * \param[in] partialResult Partial result obtained in the first step of the distributed algorithm
 */
void DistributedInput<step2Master>::add(MasterInputId id, const SharedPtr<PartialResult> &partialResult)
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
void DistributedInput<step2Master>::set(MasterInputId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}
/**
 * Returns the collection of input objects
 * \param[in] id   Identifier of the input object, \ref MasterInputId
 * \return Collection of distributed input objects
 */
DataCollectionPtr DistributedInput<step2Master>::get(MasterInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));
}
/**
 * Checks algorithm parameters on the master node
 * \param[in] parameter Pointer to the algorithm parameters
 * \param[in] method    Computation method
 */
void DistributedInput<step2Master>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    DataCollectionPtr collectionPtr = DataCollection::cast(Argument::get(0));

    if(!collectionPtr) { this->_errors->add(ErrorNullInputDataCollection); return; }
    size_t nBlocks = collectionPtr->size();
    if(collectionPtr->size() == 0) { this->_errors->add(ErrorIncorrectNumberOfInputNumericTables); return; }

    for(size_t j = 0; j < nBlocks; j++)
    {
        PartialResultPtr partialResult = PartialResult::cast((*collectionPtr)[j]);
        if(partialResult.get() == 0) { this->_errors->add(ErrorIncorrectElementInPartialResultCollection); return; }

        /* Checks partial number of observations */
        int unexpectedLayouts = (int)NumericTableIface::csrArray;
        if(!checkNumericTable(partialResult->get(nObservations).get(), this->_errors.get(), nObservationsStr(), unexpectedLayouts, 0, 1, 1)) { return; }

        unexpectedLayouts = (int)packed_mask;
        if(!checkNumericTable(partialResult->get(partialMinimum).get(), this->_errors.get(), partialMinimumStr(), unexpectedLayouts)) { return; }

        size_t nFeatures = partialResult->get(partialMinimum)->getNumberOfColumns();
        const char* errorMessages[] = {partialMinimumStr(), partialMaximumStr(), partialSumStr(), partialSumSquaresStr(), partialSumSquaresCenteredStr() };

        for(size_t i = 1; i < nPartialResults; i++)
        {
            if(!checkNumericTable(partialResult->get((PartialResultId)i).get(), this->_errors.get(), errorMessages[i-1], unexpectedLayouts, 0, nFeatures, 1)) { return; }
        }
    }
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
