/* file: truncated_gaussian_batch_fpt.cpp */
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

//++
//  Implementation of truncated gaussian initializer functions.
//--


#include "neural_networks/initializers/truncated_gaussian/truncated_gaussian_initializer.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace truncated_gaussian
{
namespace interface1
{

template<typename algorithmFPType>
DAAL_EXPORT Parameter<algorithmFPType>::Parameter(double _mean, double _sigma, size_t _seed) :
    mean(_mean), sigma(_sigma), seed(_seed)
{
    a = (algorithmFPType)(mean - 2.0 * sigma);
    b = (algorithmFPType)(mean + 2.0 * sigma);
}

template<typename algorithmFPType>
DAAL_EXPORT services::Status Parameter<algorithmFPType>::check() const
{
    DAAL_CHECK_EX(a < b, services::ErrorIncorrectParameter, services::ParameterName, aStr());
    DAAL_CHECK_EX(sigma > 0, services::ErrorIncorrectParameter, services::ParameterName, sigmaStr());
    return services::Status();
}

template DAAL_EXPORT Parameter<DAAL_FPTYPE>::Parameter(double mean, double sigma, size_t seed);
template DAAL_EXPORT services::Status Parameter<DAAL_FPTYPE>::check() const;

} // interface1
} // truncated_gaussian
} // initializers
} // neural_networks
} // algorithms
} // daal
