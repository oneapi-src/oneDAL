/* file: svd_distr_step3_input.cpp */
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
DistributedStep3Input::DistributedStep3Input() : daal::algorithms::Input(lastFinalizeOnLocalInputId + 1) {}

DistributedStep3Input::DistributedStep3Input(const DistributedStep3Input& other) : daal::algorithms::Input(other){}

/**
 * Returns input object for the SVD algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedStep3Input::get(FinalizeOnLocalInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Sets input object for the SVD algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] value Pointer to the input object value
 */
void DistributedStep3Input::set(FinalizeOnLocalInputId id, const DataCollectionPtr &value)
{
    Argument::set(id, value);
}

Status DistributedStep3Input::getSizes(size_t& nFeatures, size_t& nVectors) const
{
    DataCollectionPtr qCollection = get(inputOfStep3FromStep1);
    DataCollectionPtr rCollection = get(inputOfStep3FromStep2);
    DAAL_CHECK_EX(qCollection, ErrorNullInputDataCollection, ArgumentName, inputOfStep3FromStep2Str());
    DAAL_CHECK_EX(rCollection, ErrorNullInputDataCollection, ArgumentName, inputOfStep3FromStep1Str());

    size_t nodeSize = qCollection->size();
    DAAL_CHECK_EX(nodeSize > 0, ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, inputOfStep3FromStep2Str());
    DAAL_CHECK_EX(nodeSize == rCollection->size(), ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, inputOfStep3FromStep1Str());

    DAAL_CHECK_EX((*qCollection)[0], ErrorNullNumericTable, ArgumentName, qCollectionStr());

    NumericTablePtr numTableInQCollection = NumericTable::cast((*qCollection)[0]);
    Status s = checkNumericTable(numTableInQCollection.get(), qCollectionStr());
    if(!s) { return s; }

    nFeatures = numTableInQCollection->getNumberOfColumns();
    nVectors = numTableInQCollection->getNumberOfRows();
    return Status();
}

/**
 * Checks parameters of the algorithm
 * \param[in] parameter Pointer to the parameters
 * \param[in] method Computation method
 */
Status DistributedStep3Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    DataCollectionPtr qCollection = get(inputOfStep3FromStep1);
    DataCollectionPtr rCollection = get(inputOfStep3FromStep2);
    DAAL_CHECK_EX(qCollection, ErrorNullInputDataCollection, ArgumentName, inputOfStep3FromStep2Str());
    DAAL_CHECK_EX(rCollection, ErrorNullInputDataCollection, ArgumentName, inputOfStep3FromStep1Str());

    size_t nodeSize = qCollection->size();
    size_t nFeatures = 0;
    size_t nVectors = 0;
    Status s = this->getSizes(nFeatures, nVectors);
    if(!s) { return s; }
    for(size_t i = 0 ; i < nodeSize ; i++)
    {
        DAAL_CHECK_EX((*qCollection)[i], ErrorNullNumericTable, ArgumentName, qCollectionStr());
        DAAL_CHECK_EX((*rCollection)[i], ErrorNullNumericTable, ArgumentName, rCollectionStr());

        NumericTablePtr numTableInQCollection = NumericTable::cast((*qCollection)[i]);
        NumericTablePtr numTableInRCollection = NumericTable::cast((*rCollection)[i]);

        DAAL_CHECK_EX(numTableInQCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, inputOfStep3FromStep2Str());
        DAAL_CHECK_EX(numTableInRCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, inputOfStep3FromStep1Str());

        int unexpectedLayouts = (int)packed_mask;
        s |= checkNumericTable(numTableInQCollection.get(), qCollectionStr(), unexpectedLayouts, 0, nFeatures);
        if(!s) { return s; }
        DAAL_CHECK_EX(numTableInQCollection->getNumberOfRows() >= nFeatures, ErrorNullNumericTable, ArgumentName, rCollectionStr());
        s |= checkNumericTable(numTableInRCollection.get(), rCollectionStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
        if(!s) { return s; }
    }
    return Status();
}

} // namespace interface1
} // namespace svd
} // namespace algorithm
} // namespace daal
