/* file: svd_dense_default_online.h */
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
#ifndef __SVD_DENSE_DEFAULT_ONLINE__
#define __SVD_DENSE_DEFAULT_ONLINE__

#include "algorithms/svd/svd_types.h"
#include "src/services/daal_strings.h"

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
using namespace daal::services;

/**
 * Allocates memory to store final results of the SVD algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
Status OnlinePartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    auto pDataCollection13 = new DataCollection();
    auto pDataCollection12 = new DataCollection();
    DAAL_CHECK_MALLOC(pDataCollection13 && pDataCollection12)
    set(outputOfStep1ForStep3, DataCollectionPtr(pDataCollection13));
    set(outputOfStep1ForStep2, DataCollectionPtr(pDataCollection12));
    return Status();
}

template <typename algorithmFPType>
Status OnlinePartialResult::initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    get(outputOfStep1ForStep3)->clear();
    get(outputOfStep1ForStep2)->clear();
    return Status();
}

/**
 * Allocates additional memory to store partial results of the SVD algorithm for each subsequent compute() method
 * \tparam     algorithmFPType    Data type to use for storage in the resulting HomogenNumericTable
 * \param[in]  m    Number of columns in the input data set
 * \param[in]  n    Number of rows in the input data set
 * \param[in]  par  Reference to the object with the algorithm parameters
 */
template <typename algorithmFPType>
Status OnlinePartialResult::addPartialResultStorage(size_t m, size_t n, Parameter & par)
{
    DataCollectionPtr rCollection = staticPointerCast<DataCollection, SerializationIface>(Argument::get(outputOfStep1ForStep2));
    Status st;
    if (rCollection)
    {
        rCollection->push_back(HomogenNumericTable<algorithmFPType>::create(m, m, NumericTable::doAllocate, &st));
    }
    else
    {
        return Status(Error::create(ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep3Str()));
    }

    if (par.leftSingularMatrix != notRequired)
    {
        DataCollectionPtr qCollection = staticPointerCast<DataCollection, SerializationIface>(Argument::get(outputOfStep1ForStep3));
        if (qCollection)
        {
            qCollection->push_back(HomogenNumericTable<algorithmFPType>::create(m, n, NumericTable::doAllocate, &st));
        }
        else
        {
            return Status(Error::create(ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep3Str()));
        }
    }
    return st;
}

} // namespace interface1
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif
