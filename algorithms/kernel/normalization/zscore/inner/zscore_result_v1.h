/* file: zscore_result_v1.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#ifndef __ZSCORE_RESULT_V1_H__
#define __ZSCORE_RESULT_V1_H__

#include "zscore_types_v1.h"

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
namespace interface1
{
class ResultImpl : public DataCollection
{
public:
    DAAL_CAST_OPERATOR(ResultImpl);

    ResultImpl(const size_t n) : DataCollection(n) {}
    ResultImpl(const ResultImpl & o) : DataCollection(o) {}
    virtual ~ResultImpl() {};

    /**
    * Allocates memory to store final results of the z-score normalization algorithms
    * \param[in] input     Input objects for the z-score normalization algorithm
    *
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::Input * input);

    /**
    * Checks the correctness of the Result object
    * \param[in] in     Pointer to the input object
    *
    * \return Status of computations
    */
    services::Status check(const daal::algorithms::Input * in) const;
};

} // namespace interface1

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif // __ZSCORE_RESULT_V1_H__
