/* file: covariance_step2_distr_input.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "covariance_types.h"
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

DistributedInput<step2Master>::DistributedInput() : InputIface(lastMasterInputId + 1)
{
    Argument::set(partialResults, DataCollectionPtr(new DataCollection()));
}

size_t DistributedInput<step2Master>::getNumberOfFeatures() const
{
    DataCollectionPtr collectionOfPartialResults =
        staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));
    if(collectionOfPartialResults)
    {
        PartialResultPtr onePartialResult =
            staticPointerCast<PartialResult, SerializationIface>((*collectionOfPartialResults)[0]);
        if(onePartialResult.get() != NULL)
        {
            NumericTablePtr ntPtr = onePartialResult->get(sum);
            if(ntPtr)
            {
                return ntPtr->getNumberOfColumns();
            }
        }
    }
    return 0;
}

void DistributedInput<step2Master>::add(MasterInputId id, const PartialResultPtr &partialResult)
{
    DataCollectionPtr collection =
        staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
    collection->push_back(staticPointerCast<SerializationIface, PartialResult>(partialResult));
}

DataCollectionPtr DistributedInput<step2Master>::get(MasterInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));
}

services::Status DistributedInput<step2Master>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    DataCollectionPtr collection =
        staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));
    DAAL_CHECK_EX(collection, ErrorNullInputDataCollection, ArgumentName, partialResultsStr());


    size_t nBlocks = collection->size();
    DAAL_CHECK_EX(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialResultsStr());

    int packedLayouts = packed_mask;
    int csrLayout = (int)NumericTableIface::csrArray;
    int crossProductUnexpectedLayout = (int)NumericTableIface::csrArray |
                                       (int)NumericTableIface::upperPackedTriangularMatrix |
                                       (int)NumericTableIface::lowerPackedTriangularMatrix;

    services::Status s;
    for(size_t j = 0; j < nBlocks; j++)
    {
        PartialResultPtr partialResult =
            staticPointerCast<PartialResult, SerializationIface>((*collection)[j]);
        DAAL_CHECK_EX(partialResult, ErrorIncorrectElementInPartialResultCollection, ArgumentName, partialResultsStr());


        /* Check partial number of observations */
        NumericTable *nObservationsTable = static_cast<NumericTable *>(partialResult->get(nObservations).get());
        s |= checkNumericTable(nObservationsTable, nObservationsStr(), csrLayout, 0, 1, 1);
        if(!s) return s;

        size_t nFeatures = getNumberOfFeatures();
        /* Check partial cross-products */
        NumericTable *crossProductTable = static_cast<NumericTable *>(partialResult->get(crossProduct).get());

        s |= checkNumericTable(crossProductTable, crossProductStr(), crossProductUnexpectedLayout, 0, nFeatures, nFeatures);
        if(!s) return s;

        /* Check partial sums */
        NumericTable *sumTable = static_cast<NumericTable *>(partialResult->get(sum).get());
        s |= checkNumericTable(sumTable, sumStr(), packedLayouts, 0, nFeatures, 1);
    }
    return s;
}

}//namespace interface1
}//namespace covariance
}// namespace algorithms
}// namespace daal
