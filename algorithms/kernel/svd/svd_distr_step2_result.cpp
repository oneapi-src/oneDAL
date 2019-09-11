/* file: svd_distr_step2_result.cpp */
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
#include "serialization_utils.h"
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
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResult, SERIALIZATION_SVD_DISTRIBUTED_PARTIAL_RESULT_ID);

/** Default constructor */
DistributedPartialResult::DistributedPartialResult() : daal::algorithms::PartialResult(lastDistributedPartialResultId + 1) {}

/**
 * Returns partial result of the SVD algorithm.
 * KeyValueDataCollection under outputOfStep2ForStep3 id is structured the same as KeyValueDataCollection under
 * inputOfStep2FromStep1 id of the algorithm input
 * \param[in] id    Identifier of the partial result
 * \return          Partial result that corresponds to the given identifier
 */
KeyValueDataCollectionPtr DistributedPartialResult::get(DistributedPartialResultCollectionId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

data_management::DataCollectionPtr DistributedPartialResult::get(DistributedPartialResultCollectionId id, size_t idx) const
{
    return services::dynamicPointerCast<data_management::DataCollection, data_management::SerializationIface>( (*get(id))[idx] );
}

/**
 * Returns the result of the SVD algorithm with the matrix R calculated
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
ResultPtr DistributedPartialResult::get(DistributedPartialResultId id) const
{
    return staticPointerCast<Result, SerializationIface>(Argument::get(id));
}

/**
 * Sets KeyValueDataCollection to store partial result of the SVD algorithm
 * \param[in] id    Identifier of partial result
 * \param[in] value Pointer to the Result object
 */
void DistributedPartialResult::set(DistributedPartialResultCollectionId id, const KeyValueDataCollectionPtr &value)
{
    Argument::set(id, value);
}

/**
 * Sets Result object to store the result of the SVD algorithm
 * \param[in] id    Identifier of the result
 * \param[in] value Pointer to the Result object
 */
void DistributedPartialResult::set(DistributedPartialResultId id, const ResultPtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks partial results of the algorithm
 * \param[in] parameter Pointer to parameters
 * \param[in] method Computation method
 */
Status DistributedPartialResult::check(const daal::algorithms::Parameter *parameter, int method) const
{
    // check key-value dataCollection;
    KeyValueDataCollectionPtr resultKeyValueDC = get(outputOfStep2ForStep3);
    DAAL_CHECK_EX(resultKeyValueDC, ErrorNullOutputDataCollection, ArgumentName, outputOfStep2ForStep3Str());

    size_t nNodes = resultKeyValueDC->size();
    DAAL_CHECK_EX(nNodes > 0, ErrorIncorrectNumberOfElementsInResultCollection, ArgumentName, outputOfStep2ForStep3Str());

    // check 1st dataCollection in key-value dataCollection;
    DAAL_CHECK_EX((*resultKeyValueDC).getValueByIndex(0), ErrorNullOutputDataCollection, ArgumentName, SVDNodeCollectionStr());
    DataCollectionPtr firstNodeCollection = DataCollection::cast((*resultKeyValueDC).getValueByIndex(0));
    DAAL_CHECK_EX(firstNodeCollection, ErrorIncorrectElementInPartialResultCollection, ArgumentName, outputOfStep2ForStep3Str());
    size_t firstNodeSize = firstNodeCollection->size();
    DAAL_CHECK_EX(firstNodeSize > 0, ErrorIncorrectNumberOfElementsInResultCollection, ArgumentName, SVDNodeCollectionStr());

    // check 1st NT in 1st dataCollection;
    DAAL_CHECK_EX((*firstNodeCollection)[0], ErrorNullNumericTable, ArgumentName, SVDNodeCollectionNTStr());
    NumericTablePtr firstNumTableInFirstNodeCollection = NumericTable::cast((*firstNodeCollection)[0]);
    DAAL_CHECK_EX(firstNumTableInFirstNodeCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, SVDNodeCollectionStr());

    Status s = checkNumericTable(firstNumTableInFirstNodeCollection.get(), SVDNodeCollectionNTStr());
    if(!s) { return s; }
    size_t nFeatures = firstNumTableInFirstNodeCollection->getNumberOfColumns();
    // check all dataCollection in key-value dataCollection
    for(size_t i = 0 ; i < nNodes ; i++)
    {
        DAAL_CHECK_EX((*resultKeyValueDC).getValueByIndex((int)i), ErrorNullOutputDataCollection, ArgumentName, SVDNodeCollectionStr());
        DataCollectionPtr nodeCollection = DataCollection::cast((*resultKeyValueDC).getValueByIndex((int)i));
        DAAL_CHECK_EX(nodeCollection, ErrorIncorrectElementInPartialResultCollection, ArgumentName, outputOfStep2ForStep3Str());
        size_t nodeSize = nodeCollection->size();
        DAAL_CHECK_EX(nodeSize > 0, ErrorIncorrectNumberOfElementsInResultCollection, ArgumentName, SVDNodeCollectionStr());
        // check all numeric tables in dataCollection
        for(size_t j = 0 ; j < nodeSize ; j++)
        {
            DAAL_CHECK_EX((*nodeCollection)[j], ErrorNullNumericTable, ArgumentName, SVDNodeCollectionNTStr());
            NumericTablePtr rNumTableInNodeCollection = NumericTable::cast((*nodeCollection)[j]);
            DAAL_CHECK_EX(rNumTableInNodeCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, SVDNodeCollectionStr());
            int unexpectedLayouts = (int)packed_mask;
            s |= checkNumericTable(rNumTableInNodeCollection.get(), SVDNodeCollectionNTStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
            if(!s) { return s; }
        }
    }
    Parameter *svdPar   = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter  ));
    int unexpectedLayouts = (int)packed_mask;
    s |= checkNumericTable(get(finalResultFromStep2Master)->get(singularValues).get(), singularValuesStr(), unexpectedLayouts, 0, nFeatures, 1);
    if(!s) { return s; }
    if(svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        if(get(finalResultFromStep2Master))
        {
            s |= checkNumericTable(get(finalResultFromStep2Master)->get(rightSingularMatrix).get(), rightSingularMatrixStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
            if(!s) { return s; }
        }
    }
    return Status();
}

/**
 * Checks final results of the algorithm
 * \param[in] input  Pointer to input objects
 * \param[in] par    Pointer to parameters
 * \param[in] method Computation method
 */
Status DistributedPartialResult::check(const daal::algorithms::Input* input, const daal::algorithms::Parameter *parameter, int method) const
{
    return check(parameter, method);
}

} // namespace interface1
} // namespace svd
} // namespace algorithm
} // namespace daal
