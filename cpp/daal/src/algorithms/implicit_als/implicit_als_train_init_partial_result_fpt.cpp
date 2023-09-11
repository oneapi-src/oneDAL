/* file: implicit_als_train_init_partial_result_fpt.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_training_init_types.h"
#include "src/algorithms/implicit_als/implicit_als_train_init_parameter.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
template <typename algorithmFPType>
DAAL_EXPORT Status PartialResultBase::allocate(size_t nParts)
{
    KeyValueDataCollectionPtr outputCollection(new KeyValueDataCollection());
    KeyValueDataCollectionPtr offsetsCollection(new KeyValueDataCollection());
    Status st;
    for (size_t i = 0; i < nParts; i++)
    {
        (*outputCollection)[i]  = HomogenNumericTable<int>::create(NULL, 1, 0, &st);
        (*offsetsCollection)[i] = HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &st);
    }
    set(outputOfInitForComputeStep3, outputCollection);
    set(offsets, offsetsCollection);
    return st;
}

template <typename algorithmFPType>
DAAL_EXPORT Status PartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step1Local> * algInput = static_cast<const DistributedInput<step1Local> *>(input);
    const DistributedParameter * algParameter     = static_cast<const DistributedParameter *>(parameter);
    implicit_als::Parameter modelParameter(algParameter->nFactors);

    Status s;
    set(partialModel, PartialModel::create<algorithmFPType>(modelParameter, algInput->getNumberOfItems(), &s));
    DAAL_CHECK_STATUS_VAR(s);

    SharedPtr<HomogenNumericTable<int> > partitionTable = internal::getPartition(algParameter, s);
    if (!s) return s;

    DAAL_CHECK(partitionTable, ErrorNullNumericTable);
    const size_t nParts = partitionTable->getNumberOfRows() - 1;
    int * partitionData = partitionTable->getArray();

    DAAL_CHECK_STATUS(s, this->PartialResultBase::allocate<algorithmFPType>(nParts));

    KeyValueDataCollectionPtr dataPartsCollection(new KeyValueDataCollection());
    for (size_t i = 0; i < nParts; i++)
    {
        (*dataPartsCollection)[i] =
            CSRNumericTable::create((algorithmFPType *)NULL, NULL, NULL, algInput->get(data)->getNumberOfRows(),
                                    (size_t)partitionData[i + 1] - partitionData[i], CSRNumericTableIface::CSRIndexing::oneBased, &s);
    }
    set(outputOfStep1ForStep2, dataPartsCollection);
    return s;
}

template <typename algorithmFPType>
DAAL_EXPORT Status DistributedPartialResultStep2::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                           const int method)
{
    const DistributedInput<step2Local> * algInput = static_cast<const DistributedInput<step2Local> *>(input);
    KeyValueDataCollectionPtr dataPartsCollection = algInput->get(inputOfStep2FromStep1);
    size_t nParts                                 = dataPartsCollection->size();

    Status s;
    DAAL_CHECK_STATUS(s, this->PartialResultBase::allocate<algorithmFPType>(nParts));

    size_t fullNItems = 0;
    for (size_t i = 0; i < nParts; i++)
    {
        fullNItems += NumericTable::cast((*dataPartsCollection)[i])->getNumberOfColumns();
    }
    set(transposedData,
        CSRNumericTable::create((algorithmFPType *)NULL, NULL, NULL, fullNItems, NumericTable::cast((*dataPartsCollection)[0])->getNumberOfRows(),
                                CSRNumericTableIface::CSRIndexing::oneBased, &s));
    return s;
}

template DAAL_EXPORT Status PartialResultBase::allocate<DAAL_FPTYPE>(size_t nParts);
template DAAL_EXPORT Status PartialResult::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                                 const int method);
template DAAL_EXPORT Status DistributedPartialResultStep2::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);

} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
