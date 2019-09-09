/* file: gaussian_initializer_task_descriptor.h */
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

#ifndef __GAUSSIAN_INITIALIZER_TASK_DESCRIPTOR_H__
#define __GAUSSIAN_INITIALIZER_TASK_DESCRIPTOR_H__

#include "neural_networks/initializers/gaussian/gaussian_initializer.h"
#include "neural_networks/initializers/gaussian/gaussian_initializer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace gaussian
{
namespace internal
{

class GaussianInitializerTaskDescriptor
{
public:
    GaussianInitializerTaskDescriptor(Result *re, Parameter *pa);

    engines::BatchBase          *engine;
    data_management::Tensor     *result;
    layers::forward::LayerIface *layer;
    double a;
    double sigma;
};

} // internal
} // gaussian
} // initializers
} // neural_networks
} // algorithms
} // daal

#endif
