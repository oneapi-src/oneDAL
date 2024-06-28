/* file: svd_dense_default_distr_step2.h */
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
//  Implementation of svd algorithm and types methods.
//--
*/
#ifndef __SVD_DENSE_DEFAULT_DISTR_STEP2__
#define __SVD_DENSE_DEFAULT_DISTR_STEP2__

#include "algorithms/svd/svd_types.h"
#include "src/services/service_data_utils.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{
/**
 * Allocates memory to store partial results of the SVD algorithm
 */
template <typename algorithmFPType>
Status DistributedPartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    set(outputOfStep2ForStep3, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
    Argument::set(finalResultFromStep2Master, ResultPtr(new Result()));
    KeyValueDataCollectionPtr inCollection = static_cast<const DistributedStep2Input *>(input)->get(inputOfStep2FromStep1);
    size_t nBlocks                         = 0;
    return setPartialResultStorage<algorithmFPType>(inCollection.get(), nBlocks);
}

/**
 * Allocates memory to store partial results of the SVD algorithm based on the known structure of partial results from step 1 in the
 * distributed processing mode.
 * KeyValueDataCollection under outputOfStep2ForStep3 id is structured the same as KeyValueDataCollection under
 * inputOfStep2FromStep1 id of the algorithm input
 * \tparam     algorithmFPType Data type to use for storage in the resulting HomogenNumericTable
 * \param[in]  inCollection    KeyValueDataCollection of all partial results from the first step of  the SVD algorithm in the distributed
 *                             processing mode
 * \param[out] nBlocks         Number of rows in the input data set
 */
template <typename algorithmFPType>
Status DistributedPartialResult::setPartialResultStorage(KeyValueDataCollection * inCollection, size_t & nBlocks)
{
    KeyValueDataCollectionPtr partialCollection = staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(outputOfStep2ForStep3));
    if (!partialCollection)
    {
        return Status();
    }

    ResultPtr result = staticPointerCast<Result, SerializationIface>(Argument::get(finalResultFromStep2Master));

    const size_t inSize = inCollection->size();
    DAAL_CHECK(inSize <= services::internal::MaxVal<int>::get(), ErrorIncorrectNumberOfElementsInInputCollection)

    DataCollection * fisrtNodeCollection = static_cast<DataCollection *>((*inCollection).getValueByIndex(0).get());
    NumericTable * firstNumericTable     = static_cast<NumericTable *>((*fisrtNodeCollection)[0].get());

    size_t m = firstNumericTable->getNumberOfColumns();
    if (result->get(singularValues).get() == nullptr)
    {
        Status s = result->allocateImpl<algorithmFPType>(m, 0);
        DAAL_CHECK_STATUS_VAR(s)
    }

    nBlocks = 0;
    Status st;
    for (size_t i = 0; i < inSize; i++)
    {
        DataCollection * nodeCollection = static_cast<DataCollection *>((*inCollection).getValueByIndex((int)i).get());
        size_t nodeKey                  = (*inCollection).getKeyByIndex((int)i);
        size_t nodeSize                 = nodeCollection->size();
        nBlocks += nodeSize;

        DataCollectionPtr nodePartialResult(new DataCollection());
        DAAL_CHECK_MALLOC(nodePartialResult)
        for (size_t j = 0; j < nodeSize; j++)
        {
            nodePartialResult->push_back(HomogenNumericTable<algorithmFPType>::create(m, m, NumericTable::doAllocate, &st));
        }
        (*partialCollection)[nodeKey] = nodePartialResult;
    }
    return st;
}

} // namespace interface1
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif
