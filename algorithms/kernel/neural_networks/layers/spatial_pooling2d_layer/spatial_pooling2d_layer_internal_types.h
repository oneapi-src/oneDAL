/* file: spatial_pooling2d_layer_internal_types.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
    static void setParameter(const pooling2d::Parameter &src, pooling2d::Parameter &dst)
    {
        dst.indices = src.indices;
        dst.kernelSizes = src.kernelSizes;
        dst.strides = src.strides;
        dst.paddings = src.paddings;
    }
};

}
}
}
}
}
}

#endif
