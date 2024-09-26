/* file: zscore_fpt.cpp */
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
//  Implementation of zscore algorithm and types methods.
//--
*/

#include "src/algorithms/normalization/zscore/zscore_result.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace interface2
{
/**
 * Allocates memory to store final results of the z-score normalization algorithms
 * \param[in] input     Input objects for the z-score normalization algorithm
 * \param[in] parameter Pointer to algorithm parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl, ErrorNullPtr);
    return impl->allocate<algorithmFPType>(input, parameter);
}

/**
* Allocates memory to store final results of the z-score normalization algorithms for interface1 API compartibility
* \param[in] input     Input objects for the z-score normalization algorithm
* \param[in] method    Algorithm computation method
*/
template <typename algorithmFPType>
Status Result::allocate(const daal::algorithms::Input * input, const int method)
{
    return allocate<algorithmFPType>(input, NULL, method);
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);
template Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const int method);

} // namespace interface2
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
