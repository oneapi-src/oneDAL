/* file: qr_dense_default_online.h */
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
#ifndef __QR_DENSE_DEFAULT_ONLINE__
#define __QR_DENSE_DEFAULT_ONLINE__

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
 * \param[in] input     Pointer to input object
 * \param[in] parameter Pointer to parameter
 * \param[in] method    Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT void OnlinePartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    set(outputOfStep1ForStep3, data_management::DataCollectionPtr(new data_management::DataCollection()));
    set(outputOfStep1ForStep2, data_management::DataCollectionPtr(new data_management::DataCollection()));
}

/**
 * Allocates additional memory for storing partial results of the QR decomposition algorithm for each subsequent call to compute method
 * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
 * \param[in]  m  Number of columns in the input data set
 * \param[in]  n  Number of rows in the input data set
 */
template <typename algorithmFPType>
DAAL_EXPORT void OnlinePartialResult::addPartialResultStorage(size_t m, size_t n)
{
    data_management::DataCollectionPtr qCollection = get(outputOfStep1ForStep3);
    data_management::DataCollectionPtr rCollection = get(outputOfStep1ForStep2);

    if(qCollection)
    {
        qCollection->push_back(data_management::SerializationIfacePtr(
                               new data_management::HomogenNumericTable<algorithmFPType>(m, n, data_management::NumericTable::doAllocate)));
    }
    else
    {
        this->_errors->add(services::Error::create(services::ErrorNullOutputDataCollection, services::ArgumentName, "msg 3 add later"));
        return;
    }
    if(rCollection)
    {
        rCollection->push_back(data_management::SerializationIfacePtr(
                               new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));
    }
    else
    {
        this->_errors->add(services::Error::create(services::ErrorNullOutputDataCollection, services::ArgumentName, "msg 3 add later"));
        return;
    }
}

}// namespace interface1
}// namespace qr
}// namespace algorithms
}// namespace daal

#endif
