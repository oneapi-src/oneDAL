/* file: qr_distr_step2_input.cpp */
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
//  Implementation of qr classes.
//--
*/

#include "algorithms/qr/qr_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace interface1
{

/** Default constructor */
DistributedStep2Input::DistributedStep2Input() : daal::algorithms::Input(1)
{
    Argument::set(inputOfStep2FromStep1,
                  data_management::KeyValueDataCollectionPtr(new data_management::KeyValueDataCollection()));
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
void DistributedStep2Input::set(MasterInputId id, const KeyValueDataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns input object for the QR decomposition algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
KeyValueDataCollectionPtr DistributedStep2Input::get(MasterInputId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Adds input object to KeyValueDataCollection  of the QR decomposition algorithm
 * \param[in] id    Identifier of input object
 * \param[in] key   Key to use to retrieve data
 * \param[in] value Pointer to the input object value
 */
void DistributedStep2Input::add(MasterInputId id, size_t key, const DataCollectionPtr &value)
{
    KeyValueDataCollectionPtr collection = staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
    (*collection)[key] = value;
}

/**
* Returns the number of blocks in the input data set
* \return Number of blocks in the input data set
*/
size_t DistributedStep2Input::getNBlocks()
{
    KeyValueDataCollectionPtr kvDC = get(inputOfStep2FromStep1);
    size_t nNodes = kvDC->size();
    size_t nBlocks = 0;
    for(size_t i = 0 ; i < nNodes ; i++)
    {
        DataCollectionPtr nodeCollection = staticPointerCast<DataCollection, SerializationIface>((*kvDC).getValueByIndex((int)i));
        size_t nodeSize = nodeCollection->size();
        nBlocks += nodeSize;
    }
    return nBlocks;
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t DistributedStep2Input::getNumberOfColumns() const
{
    KeyValueDataCollectionPtr inputKeyValueDC = get(inputOfStep2FromStep1);
    // check key-value dataCollection;
    if(!inputKeyValueDC)
    {
        this->_errors->add(Error::create(ErrorNullInputDataCollection, ArgumentName, inputOfStep2FromStep1Str()));
        return 0;
    }
    size_t nNodes = inputKeyValueDC->size();
    if(nNodes == 0)
    {
        this->_errors->add(Error::create(ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, inputOfStep2FromStep1Str()));
        return 0;
    }
    // check 1st dataCollection in key-value dataCollection;
    if(!(*inputKeyValueDC).getValueByIndex(0))
    {
        this->_errors->add(Error::create(ErrorNullInputDataCollection, ArgumentName, QRNodeCollectionStr()));
        return 0;
    }
    DataCollectionPtr firstNodeCollection = DataCollection::cast((*inputKeyValueDC).getValueByIndex(0));
    if(!firstNodeCollection)
    {
        this->_errors->add(Error::create(ErrorIncorrectElementInPartialResultCollection, ArgumentName, inputOfStep2FromStep1Str()));
        return 0;
    }
    size_t firstNodeSize = firstNodeCollection->size();
    if(firstNodeSize == 0)
    {
        this->_errors->add(Error::create(ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, QRNodeCollectionStr()));
        return 0;
    }
    // check 1st NT in 1st dataCollection;
    if(!(*firstNodeCollection)[0])
    {
        this->_errors->add(Error::create(ErrorNullNumericTable, ArgumentName, QRNodeCollectionNTStr()));
        return 0;
    }
    NumericTablePtr firstNumTableInFirstNodeCollection = NumericTable::cast((*firstNodeCollection)[0]);
    if(!firstNumTableInFirstNodeCollection)
    {
        this->_errors->add(Error::create(ErrorIncorrectElementInNumericTableCollection, ArgumentName, QRNodeCollectionStr()));
        return 0;
    }
    if(!checkNumericTable(firstNumTableInFirstNodeCollection.get(), this->_errors.get(), QRNodeCollectionNTStr())) { return 0; }
    return firstNumTableInFirstNodeCollection->getNumberOfColumns();
}

/**
 * Checks parameters of the algorithm
 * \param[in] parameter Pointer to the parameters
 * \param[in] method Computation method
 */
void DistributedStep2Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    // check key-value dataCollection;
    KeyValueDataCollectionPtr inputKeyValueDC = get(inputOfStep2FromStep1);
    size_t nFeatures = getNumberOfColumns();
    if(nFeatures == 0)
    {
        return;
    }
    size_t nNodes = inputKeyValueDC->size();
    // check all dataCollection in key-value dataCollection
    for(size_t i = 0 ; i < nNodes ; i++)
    {
        DAAL_CHECK_EX((*inputKeyValueDC).getValueByIndex((int)i), ErrorNullInputDataCollection, ArgumentName, QRNodeCollectionStr());
        DataCollectionPtr nodeCollection = DataCollection::cast((*inputKeyValueDC).getValueByIndex((int)i));
        DAAL_CHECK_EX(nodeCollection, ErrorIncorrectElementInPartialResultCollection, ArgumentName, inputOfStep2FromStep1Str());
        size_t nodeSize = nodeCollection->size();
        DAAL_CHECK_EX(nodeSize > 0, ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, QRNodeCollectionStr());

        // check all numeric tables in dataCollection
        for(size_t j = 0 ; j < nodeSize ; j++)
        {
            DAAL_CHECK_EX((*nodeCollection)[j], ErrorNullNumericTable, ArgumentName, QRNodeCollectionNTStr());
            NumericTablePtr numTableInNodeCollection = NumericTable::cast((*nodeCollection)[j]);
            DAAL_CHECK_EX(numTableInNodeCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, QRNodeCollectionStr());
            int unexpectedLayouts = (int)packed_mask;
            if(!checkNumericTable(numTableInNodeCollection.get(), this->_errors.get(), QRNodeCollectionNTStr(), unexpectedLayouts, 0, nFeatures, nFeatures)) { return; }
        }
    }
}

} // namespace interface1
} // namespace qr
} // namespace algorithm
} // namespace daal
