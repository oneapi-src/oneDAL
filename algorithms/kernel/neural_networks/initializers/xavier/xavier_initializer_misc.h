/* file: xavier_initializer_misc.h */
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
//  Implementation of service functions for xavier initializer
//--
*/

#ifndef __XAVIER_INITIALIZER_MISC_H__
#define __XAVIER_INITIALIZER_MISC_H__

#include "neural_networks/initializers/xavier/xavier_initializer.h"
#include "neural_networks/initializers/xavier/xavier_initializer_types.h"
#include "xavier_initializer_task_descriptor.h"

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
namespace internal
{

template<typename algorithmFPType>
services::Status getFanInAndFanOut(const XavierInitializerTaskDescriptor &desc,
                                   size_t &fanIn, size_t &fanOut);

} // internal
} // xavier
} // initializers
} // neural_networks
} // algorithms
} // daal

#endif
