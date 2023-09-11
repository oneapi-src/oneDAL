/* file: svd_distr_step2_input.cpp */
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
//  Implementation of svd classes.
//--
*/

#include "algorithms/svd/svd_types.h"
#include "src/services/daal_strings.h"
#include "src/services/service_data_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
/** Default constructor */
DistributedStep2Input::DistributedStep2Input() : daal::algorithms::Input(lastMasterInputId + 1)
{
    Argument::set(inputOfStep2FromStep1, data_management::KeyValueDataCollectionPtr(new data_management::KeyValueDataCollection()));
}

/** Copy constructor */
DistributedStep2Input::DistributedStep2Input(const DistributedStep2Input & other) : daal::algorithms::Input(other) {}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
void DistributedStep2Input::set(MasterInputId id, const KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns input object for the SVD algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
KeyValueDataCollectionPtr DistributedStep2Input::get(MasterInputId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Adds input object to KeyValueDataCollection  of the SVD algorithm
 * \param[in] id    Identifier of input object
 * \param[in] key   Key to use to retrieve data
 * \param[in] value Pointer to the input object value
 */
void DistributedStep2Input::add(MasterInputId id, size_t key, const DataCollectionPtr & value)
{
    KeyValueDataCollectionPtr collection = staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
    (*collection)[key]                   = value;
}

/**
* Returns the number of blocks in the input data set
* \return Number of blocks in the input data set
*/
size_t DistributedStep2Input::getNBlocks()
{
    KeyValueDataCollectionPtr kvDC = get(inputOfStep2FromStep1);
    size_t nNodes                  = kvDC->size();
    DAAL_ASSERT(nNodes <= services::internal::MaxVal<int>::get())
    size_t nBlocks = 0;
    for (size_t i = 0; i < nNodes; i++)
    {
        DataCollectionPtr nodeCollection = staticPointerCast<DataCollection, SerializationIface>((*kvDC).getValueByIndex((int)i));
        size_t nodeSize                  = nodeCollection->size();
        nBlocks += nodeSize;
    }
    return nBlocks;
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
Status DistributedStep2Input::getNumberOfColumns(size_t & nCols) const
{
    nCols                                     = 0;
    KeyValueDataCollectionPtr inputKeyValueDC = get(inputOfStep2FromStep1);
    // check key-value dataCollection;
    DAAL_CHECK_EX(inputKeyValueDC, ErrorNullInputDataCollection, ArgumentName, inputOfStep2FromStep1Str());

    size_t nNodes = inputKeyValueDC->size();
    DAAL_CHECK_EX(nNodes != 0, ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, inputOfStep2FromStep1Str());

    // check 1st dataCollection in key-value dataCollection;
    DAAL_CHECK_EX((*inputKeyValueDC).getValueByIndex(0), ErrorNullInputDataCollection, ArgumentName, SVDNodeCollectionStr());

    DataCollectionPtr firstNodeCollection = DataCollection::cast((*inputKeyValueDC).getValueByIndex(0));
    DAAL_CHECK_EX(firstNodeCollection, ErrorIncorrectElementInPartialResultCollection, ArgumentName, SVDNodeCollectionStr());

    size_t firstNodeSize = firstNodeCollection->size();
    DAAL_CHECK_EX(firstNodeSize != 0, ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, SVDNodeCollectionStr());

    // check 1st NT in 1st dataCollection;
    DAAL_CHECK_EX((*firstNodeCollection)[0], ErrorNullNumericTable, ArgumentName, SVDNodeCollectionNTStr());

    NumericTablePtr firstNumTableInFirstNodeCollection = NumericTable::cast((*firstNodeCollection)[0]);
    DAAL_CHECK_EX(firstNumTableInFirstNodeCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, SVDNodeCollectionStr());

    Status s = checkNumericTable(firstNumTableInFirstNodeCollection.get(), SVDNodeCollectionNTStr());
    if (!s)
    {
        return s;
    }

    nCols = firstNumTableInFirstNodeCollection->getNumberOfColumns();
    return Status();
}

/**
 * Checks parameters of the algorithm
 * \param[in] parameter Pointer to the parameters
 * \param[in] method Computation method
 */
Status DistributedStep2Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    // check key-value dataCollection;
    KeyValueDataCollectionPtr inputKeyValueDC = get(inputOfStep2FromStep1);
    size_t nFeatures                          = 0;
    Status s                                  = getNumberOfColumns(nFeatures);
    if (!s)
    {
        return s;
    }
    if (nFeatures == 0)
    {
        return s;
    }

    const size_t nNodes = inputKeyValueDC->size();
    DAAL_CHECK(nNodes <= services::internal::MaxVal<int>::get(), ErrorIncorrectNumberOfNodes)
    // check all dataCollection in key-value dataCollection
    for (size_t i = 0; i < nNodes; i++)
    {
        DAAL_CHECK_EX((*inputKeyValueDC).getValueByIndex((int)i), ErrorNullInputDataCollection, ArgumentName, SVDNodeCollectionStr());
        DataCollectionPtr nodeCollection = DataCollection::cast((*inputKeyValueDC).getValueByIndex((int)i));
        DAAL_CHECK_EX(nodeCollection, ErrorIncorrectElementInPartialResultCollection, ArgumentName, inputOfStep2FromStep1Str());
        size_t nodeSize = nodeCollection->size();
        DAAL_CHECK_EX(nodeSize > 0, ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, SVDNodeCollectionStr());

        // check all numeric tables in dataCollection
        for (size_t j = 0; j < nodeSize; j++)
        {
            DAAL_CHECK_EX((*nodeCollection)[j], ErrorNullNumericTable, ArgumentName, SVDNodeCollectionNTStr());
            NumericTablePtr numTableInNodeCollection = NumericTable::cast((*nodeCollection)[j]);
            DAAL_CHECK_EX(numTableInNodeCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, SVDNodeCollectionStr());
            int unexpectedLayouts = (int)packed_mask;
            s |= checkNumericTable(numTableInNodeCollection.get(), SVDNodeCollectionNTStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
            if (!s)
            {
                return s;
            }
        }
    }
    return Status();
}

} // namespace svd
} // namespace algorithms
} // namespace daal
