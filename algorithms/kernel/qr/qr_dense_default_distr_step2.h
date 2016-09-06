/* file: qr_dense_default_distr_step2.h */
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
//  Implementation of qr algorithm and types methods.
//--
*/
#ifndef __QR_DENSE_DEFAULT_DISTR_STEP2__
#define __QR_DENSE_DEFAULT_DISTR_STEP2__

#include "qr_types.h"

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace interface1
{

/**
 * Allocates memory for storing partial results of the QR decomposition algorithm
 * \param[in] input  Pointer to input object
 * \param[in] parameter    Pointer to parameter
 * \param[in] method Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void DistributedPartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    Argument::set(outputOfStep2ForStep3,
                  data_management::KeyValueDataCollectionPtr(new data_management::KeyValueDataCollection()));
    Argument::set(finalResultFromStep2Master, services::SharedPtr<Result>(new Result()));
    data_management::KeyValueDataCollectionPtr inCollection = static_cast<const DistributedStep2Input *>(input)->get(
                                                                                    inputOfStep2FromStep1);
    size_t nBlocks = 0;
    setPartialResultStorage<algorithmFPType>(inCollection.get(), nBlocks);
}

/**
 * Allocates memory for storing partial results of the QR decomposition algorithm based on known structure of partial results from the
 * first steps of the algorithm in the distributed processing mode.
 * KeyValueDataCollection under outputOfStep2ForStep3 is structured the same as KeyValueDataCollection under
 * inputOfStep2FromStep1 id of the algorithm input
 * \tparam     algorithmFPType             Data type to be used for storage in resulting HomogenNumericTable
 * \param[in]  inCollection  KeyValueDataCollection of all partial results from the first steps of the algorithm in the distributed
 * processing mode
 * \param[out] nBlocks  Number of rows in the input data set
 */
template <typename algorithmFPType>
DAAL_EXPORT void DistributedPartialResult::setPartialResultStorage(data_management::KeyValueDataCollection *inCollection, size_t &nBlocks)
{
    data_management::KeyValueDataCollectionPtr partialCollection =
        services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(
                                                                                                                      outputOfStep2ForStep3));
    if(!partialCollection)
    {
        return;
    }

    services::SharedPtr<Result> result =
        services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(finalResultFromStep2Master));

    size_t inSize = inCollection->size();

    data_management::DataCollection *fisrtNodeCollection =
        static_cast<data_management::DataCollection *>((*inCollection).getValueByIndex(0).get());
    data_management::NumericTable   *firstNumericTable   = static_cast<data_management::NumericTable *>((*fisrtNodeCollection)[0].get());

    size_t m = firstNumericTable->getNumberOfColumns();
    if(result->get(matrixR).get() == NULL)
    {
        result->allocateImpl<algorithmFPType>(m, 0);
    }

    nBlocks = 0;
    for(size_t i = 0 ; i < inSize ; i++)
    {
        data_management::DataCollection   *nodeCollection
            = static_cast<data_management::DataCollection *>((*inCollection).getValueByIndex((int)i).get());
        size_t            nodeKey        = (*inCollection).getKeyByIndex((int)i);
        size_t nodeSize = nodeCollection->size();
        nBlocks += nodeSize;

        data_management::DataCollectionPtr nodePartialResult(new data_management::DataCollection());

        for(size_t j = 0 ; j < nodeSize ; j++)
        {
            nodePartialResult->push_back(
                data_management::SerializationIfacePtr(
                    new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));
        }
        (*partialCollection)[ nodeKey ] = nodePartialResult;
    }
}

}// namespace interface1
}// namespace qr
}// namespace algorithms
}// namespace daal

#endif
