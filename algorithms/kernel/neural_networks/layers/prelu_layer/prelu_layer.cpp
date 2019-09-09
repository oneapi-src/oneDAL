/* file: prelu_layer.cpp */
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
//  Implementation of prelu calculation algorithm and types methods.
//--
*/

#include "prelu_layer_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace prelu
{
namespace interface1
{
/**
*  Constructs parameters of the prelu layer
*  \param[in] _dataDimension    Starting data dimension index to apply weight
*  \param[in] _weightsDimension Number of weight dimensions
*/
Parameter::Parameter(const size_t _dataDimension, const size_t _weightsDimension) : dataDimension(_dataDimension),
    weightsDimension(_weightsDimension)
{};

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    if(weightsDimension == (size_t)0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, weightsDimensionStr()));
    }
    return services::Status();
}

}// namespace interface1
}// namespace prelu
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
