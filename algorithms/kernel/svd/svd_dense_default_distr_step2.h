/* file: svd_dense_default_distr_step2.h */
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
//  Implementation of svd algorithm and types methods.
//--
*/
#ifndef __SVD_DENSE_DEFAULT_DISTR_STEP2__
#define __SVD_DENSE_DEFAULT_DISTR_STEP2__

#include "svd_types.h"

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
DAAL_EXPORT Status DistributedPartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    set(outputOfStep2ForStep3, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
    Argument::set(finalResultFromStep2Master, ResultPtr(new Result()));
    KeyValueDataCollectionPtr inCollection = static_cast<const DistributedStep2Input *>(input)->get(inputOfStep2FromStep1);
    size_t nBlocks = 0;
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
DAAL_EXPORT Status DistributedPartialResult::setPartialResultStorage(KeyValueDataCollection *inCollection, size_t &nBlocks)
{
    KeyValueDataCollectionPtr partialCollection =
        staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(outputOfStep2ForStep3));
    if(!partialCollection)
    {
        return Status();
    }

    ResultPtr result = staticPointerCast<Result, SerializationIface>(Argument::get(finalResultFromStep2Master));

    size_t inSize = inCollection->size();

    DataCollection *fisrtNodeCollection = static_cast<DataCollection *>((*inCollection).getValueByIndex(0).get());
    NumericTable *firstNumericTable     = static_cast<NumericTable *>((*fisrtNodeCollection)[0].get());

    size_t m = firstNumericTable->getNumberOfColumns();
    if(result->get(singularValues).get() == NULL)
    {
        Status s = result->allocateImpl<algorithmFPType>(m, 0);
        if(!s)
            return s;
    }

    nBlocks = 0;
    Status st;
    for(size_t i = 0 ; i < inSize ; i++)
    {
        DataCollection   *nodeCollection = static_cast<DataCollection *>((*inCollection).getValueByIndex((int)i).get());
        size_t            nodeKey        = (*inCollection).getKeyByIndex((int)i);
        size_t nodeSize = nodeCollection->size();
        nBlocks += nodeSize;

        DataCollectionPtr nodePartialResult(new DataCollection());
        for(size_t j = 0 ; j < nodeSize ; j++)
        {
            nodePartialResult->push_back(HomogenNumericTable<algorithmFPType>::create(m, m, NumericTable::doAllocate, &st));
        }
        (*partialCollection)[ nodeKey ] = nodePartialResult;
    }
    return st;
}

}// namespace interface1
}// namespace svd
}// namespace algorithms
}// namespace daal

#endif
