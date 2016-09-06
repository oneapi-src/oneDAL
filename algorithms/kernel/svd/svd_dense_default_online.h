/* file: svd_dense_default_online.h */
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
//  Implementation of svd algorithm and types methods.
//--
*/
#ifndef __SVD_DENSE_DEFAULT_ONLINE__
#define __SVD_DENSE_DEFAULT_ONLINE__

#include "svd_types.h"

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{

/**
 * Allocates memory to store final results of the SVD algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void OnlinePartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    Argument::set(outputOfStep1ForStep3, data_management::DataCollectionPtr(new data_management::DataCollection()));
    Argument::set(outputOfStep1ForStep2, data_management::DataCollectionPtr(new data_management::DataCollection()));
}

/**
 * Allocates additional memory to store partial results of the SVD algorithm for each subsequent compute() method
 * \tparam     algorithmFPType    Data type to use for storage in the resulting HomogenNumericTable
 * \param[in]  m    Number of columns in the input data set
 * \param[in]  n    Number of rows in the input data set
 * \param[in]  par  Reference to the object with the algorithm parameters
 */
template <typename algorithmFPType>
DAAL_EXPORT void OnlinePartialResult::addPartialResultStorage(size_t m, size_t n, Parameter &par)
{
    data_management::DataCollectionPtr rCollection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(outputOfStep1ForStep2));

    if(rCollection)
    {
        rCollection->push_back(data_management::SerializationIfacePtr(
                               new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));
    }
    else
    {
        this->_errors->add(services::Error::create(services::ErrorNullOutputDataCollection, services::ArgumentName, outputOfStep1ForStep3Str()));
        return;
    }

    if(par.leftSingularMatrix != notRequired)
    {
        data_management::DataCollectionPtr qCollection =
            services::staticPointerCast<data_management::DataCollection,
            data_management::SerializationIface>(Argument::get(outputOfStep1ForStep3));
        if(qCollection)
        {
            qCollection->push_back(data_management::SerializationIfacePtr(
                                   new data_management::HomogenNumericTable<algorithmFPType>(m, n, data_management::NumericTable::doAllocate)));
        }
        else
        {
            this->_errors->add(services::Error::create(services::ErrorNullOutputDataCollection, services::ArgumentName, outputOfStep1ForStep3Str()));
            return;
        }
    }
}

}// namespace interface1
}// namespace svd
}// namespace algorithms
}// namespace daal

#endif
