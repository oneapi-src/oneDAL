/* file: spatial_pooling2d_layer_internal_types.h */
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
//  Common classes for 2D pooling layers
//--
*/

#ifndef __SPATIAL_POOLING2D_LAYER_INTERNAL_PARAMETER_H__
#define __SPATIAL_POOLING2D_LAYER_INTERNAL_PARAMETER_H__

#include "service_utils.h"
#include "tensor.h"
#include "collection.h"
#include "service_blas.h"
#include "neural_networks/layers/pooling2d/pooling2d_layer_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_pooling2d
{
namespace internal
{

enum Method
{
    maximum = 0,
    average = 1,
    stochastic = 2
};

template<CpuType cpu>
class CommonSpatialPoolingFunctions
{
public:
    static void setParameter(const pooling2d::Parameter *src, pooling2d::Parameter *dst)
    {
        dst->indices = src->indices;
        dst->kernelSizes = src->kernelSizes;
        dst->strides = src->strides;
        dst->paddings = src->paddings;
    }
};

}
}
}
}
}
}

#endif
