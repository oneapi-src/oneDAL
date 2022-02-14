/* file: uniform_types_fpt.cpp */
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

//++
//  Implementation of uniform distribution algorithm and types methods.
//--

#include "algorithms/distributions/uniform/uniform_types.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace uniform
{
namespace interface1
{
/**
  * Check the correctness of the %Parameter object
  */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Parameter<algorithmFPType>::check() const
{
    DAAL_CHECK_EX(a < b, services::ErrorIncorrectParameter, services::ParameterName, aStr());
    return services::Status();
}

template DAAL_EXPORT services::Status Parameter<DAAL_FPTYPE>::check() const;

} // namespace interface1
} // namespace uniform
} // namespace distributions
} // namespace algorithms
} // namespace daal
