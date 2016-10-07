/* file: uniform_initializer_batch.cpp */
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

//++
//  Implementation of uniform calculation functions.
//--


#include "neural_networks/initializers/uniform/uniform_initializer.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace uniform
{

namespace interface1
{

template<typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch(double a, double b, size_t seed) : parameter(a, b, seed)
{
    initialize();
}

template class Batch<double, defaultDense>;
template class Batch<float, defaultDense>;

} // interface1



}
}
}
}
}
