/* file: xavier_initializer_batch.cpp */
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
//  Implementation of Xavier calculation functions.
//--


#include "neural_networks/initializers/xavier/xavier_initializer.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace xavier
{
namespace interface1
{

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(layer, services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, layerStr());
    return services::Status();
}

} // interface1
} // xavier
} // initializers
} // neural_networks
} // algorithms
} // daal
