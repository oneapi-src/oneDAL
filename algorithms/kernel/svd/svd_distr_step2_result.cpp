/* file: svd_distr_step2_result.cpp */
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
//  Implementation of svd classes.
//--
*/

#include "algorithms/svd/svd_types.h"

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
DistributedPartialResult::DistributedPartialResult() : daal::algorithms::PartialResult(2) {}

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

/**
 * Returns the result of the SVD algorithm with the matrix R calculated
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
SharedPtr<Result> DistributedPartialResult::get(DistributedPartialResultId id) const
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
void DistributedPartialResult::set(DistributedPartialResultId id, const SharedPtr<Result> &value)
{
    Argument::set(id, value);
}

/**
 * Checks partial results of the algorithm
 * \param[in] parameter Pointer to parameters
 * \param[in] method Computation method
 */
void DistributedPartialResult::check(const daal::algorithms::Parameter *parameter, int method) const
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

    if(!checkNumericTable(firstNumTableInFirstNodeCollection.get(), this->_errors.get(), SVDNodeCollectionNTStr())) { return; }
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
            if(!checkNumericTable(rNumTableInNodeCollection.get(), this->_errors.get(), SVDNodeCollectionNTStr(), unexpectedLayouts, 0, nFeatures, nFeatures)) { return; }
        }
    }
    Parameter *svdPar   = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter  ));
    int unexpectedLayouts = (int)packed_mask;
    if(!checkNumericTable(get(finalResultFromStep2Master)->get(singularValues).get(), this->_errors.get(), singularValuesStr(), unexpectedLayouts, 0, nFeatures, 1)) { return; }
    nFeatures = get(finalResultFromStep2Master)->get(singularValues)->getNumberOfColumns();
    if(svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        if(get(finalResultFromStep2Master))
        {
            if(!checkNumericTable(get(finalResultFromStep2Master)->get(rightSingularMatrix).get(), this->_errors.get(), rightSingularMatrixStr(),
                unexpectedLayouts, 0, nFeatures, nFeatures)) { return; }
        }
    }
}

/**
 * Checks final results of the algorithm
 * \param[in] input  Pointer to input objects
 * \param[in] par    Pointer to parameters
 * \param[in] method Computation method
 */
void DistributedPartialResult::check(const daal::algorithms::Input* input, const daal::algorithms::Parameter *parameter, int method) const
{
    check(parameter, method);
}

} // namespace interface1
} // namespace svd
} // namespace algorithm
} // namespace daal
