/* file: svd_distr_step2_input.cpp */
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
//  Implementation of svd classes.
//--
*/

#include "algorithms/svd/svd_types.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{

/** Default constructor */
DistributedStep2Input::DistributedStep2Input() : daal::algorithms::Input(lastMasterInputId + 1)
{
    Argument::set(inputOfStep2FromStep1,
                  data_management::KeyValueDataCollectionPtr(new data_management::KeyValueDataCollection()));
}

/** Copy constructor */
DistributedStep2Input::DistributedStep2Input(const DistributedStep2Input& other) : daal::algorithms::Input(other){}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
void DistributedStep2Input::set(MasterInputId id, const KeyValueDataCollectionPtr &ptr)
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
Status DistributedStep2Input::getNumberOfColumns(size_t& nCols) const
{
    nCols = 0;
    KeyValueDataCollectionPtr inputKeyValueDC = get(inputOfStep2FromStep1);
    // check key-value dataCollection;
    DAAL_CHECK_EX(inputKeyValueDC, ErrorNullInputDataCollection, ArgumentName, inputOfStep2FromStep1Str());

    size_t nNodes = inputKeyValueDC->size();
    DAAL_CHECK_EX(nNodes  != 0, ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, inputOfStep2FromStep1Str());

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
    if(!s) { return s; }

    nCols = firstNumTableInFirstNodeCollection->getNumberOfColumns();
    return Status();
}

/**
 * Checks parameters of the algorithm
 * \param[in] parameter Pointer to the parameters
 * \param[in] method Computation method
 */
Status DistributedStep2Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    // check key-value dataCollection;
    KeyValueDataCollectionPtr inputKeyValueDC = get(inputOfStep2FromStep1);
    size_t nFeatures = 0;
    Status s = getNumberOfColumns(nFeatures);
    if(!s) { return s; }
    if(nFeatures == 0) { return s; }

    size_t nNodes = inputKeyValueDC->size();
    // check all dataCollection in key-value dataCollection
    for(size_t i = 0 ; i < nNodes ; i++)
    {
        DAAL_CHECK_EX((*inputKeyValueDC).getValueByIndex((int)i), ErrorNullInputDataCollection, ArgumentName, SVDNodeCollectionStr());
        DataCollectionPtr nodeCollection = DataCollection::cast((*inputKeyValueDC).getValueByIndex((int)i));
        DAAL_CHECK_EX(nodeCollection, ErrorIncorrectElementInPartialResultCollection, ArgumentName, inputOfStep2FromStep1Str());
        size_t nodeSize = nodeCollection->size();
        DAAL_CHECK_EX(nodeSize > 0, ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, SVDNodeCollectionStr());

        // check all numeric tables in dataCollection
        for(size_t j = 0 ; j < nodeSize ; j++)
        {
            DAAL_CHECK_EX((*nodeCollection)[j], ErrorNullNumericTable, ArgumentName, SVDNodeCollectionNTStr());
            NumericTablePtr numTableInNodeCollection = NumericTable::cast((*nodeCollection)[j]);
            DAAL_CHECK_EX(numTableInNodeCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, SVDNodeCollectionStr());
            int unexpectedLayouts = (int)packed_mask;
            s |= checkNumericTable(numTableInNodeCollection.get(),
                SVDNodeCollectionNTStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
            if(!s) { return s; }
        }
    }
    return Status();
}

} // namespace interface1
} // namespace svd
} // namespace algorithm
} // namespace daal
