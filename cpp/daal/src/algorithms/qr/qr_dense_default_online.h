/* file: qr_dense_default_online.h */
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
//  Implementation of qr algorithm and types methods.
//--
*/
#ifndef __QR_DENSE_DEFAULT_ONLINE__
#define __QR_DENSE_DEFAULT_ONLINE__

#include "algorithms/qr/qr_types.h"
#include "src/services/daal_strings.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace qr
{
/**
 * Allocates memory for storing partial results of the QR decomposition algorithm
 * \param[in] input     Pointer to input object
 * \param[in] parameter Pointer to parameter
 * \param[in] method    Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status OnlinePartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                 const int method)
{
    auto pDataCollection13 = new DataCollection();
    auto pDataCollection12 = new DataCollection();
    DAAL_CHECK_MALLOC(pDataCollection13 && pDataCollection12)
    set(outputOfStep1ForStep3, DataCollectionPtr(pDataCollection13));
    set(outputOfStep1ForStep2, DataCollectionPtr(pDataCollection12));
    return Status();
}

template <typename algorithmFPType>
DAAL_EXPORT Status OnlinePartialResult::initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                   const int method)
{
    get(outputOfStep1ForStep3)->clear();
    get(outputOfStep1ForStep2)->clear();
    return Status();
}

/**
 * Allocates additional memory for storing partial results of the QR decomposition algorithm for each subsequent call to compute method
 * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
 * \param[in]  m  Number of columns in the input data set
 * \param[in]  n  Number of rows in the input data set
 */
template <typename algorithmFPType>
DAAL_EXPORT Status OnlinePartialResult::addPartialResultStorage(size_t m, size_t n)
{
    DataCollectionPtr qCollection = get(outputOfStep1ForStep3);
    DataCollectionPtr rCollection = get(outputOfStep1ForStep2);

    Status s;
    if (qCollection)
    {
        qCollection->push_back(HomogenNumericTable<algorithmFPType>::create(m, n, NumericTable::doAllocate, &s));
    }
    else
    {
        return Status(Error::create(ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep3Str()));
    }
    if (rCollection)
    {
        rCollection->push_back(HomogenNumericTable<algorithmFPType>::create(m, m, NumericTable::doAllocate, &s));
    }
    else
    {
        return Status(Error::create(ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep2Str()));
    }
    return s;
}

} // namespace qr
} // namespace algorithms
} // namespace daal

#endif
