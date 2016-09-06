/* file: pooling2d_layer_internal_parameter.h */
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

#ifndef __POOLING2D_LAYER_INTERNAL_PARAMETER_H__
#define __POOLING2D_LAYER_INTERNAL_PARAMETER_H__

#include "service_utils.h"
#include "tensor.h"
#include "collection.h"
#include "service_blas.h"

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
namespace pooling2d
{
namespace internal
{

struct Parameter
{
    /*
     * Input data tensor is viewed by this method as a 5-dimensional tensor of size:
     * offsetBefore * firstSize * offsetBetween * secondSize * offsetAfter
     */
    Parameter(const size_t *indices, const size_t *padding, const size_t *stride, const size_t *kernelSize,
              Tensor *dataTensor, const Collection<size_t> &dims, const Collection<size_t> &valueDims);

    MKL_INT firstIndex;
    MKL_INT secondIndex;
    MKL_INT firstPadding;
    MKL_INT secondPadding;
    MKL_INT firstStride;
    MKL_INT secondStride;
    MKL_INT firstKernelSize;
    MKL_INT secondKernelSize;

    MKL_INT offsetBefore;
    MKL_INT firstSize;
    MKL_INT firstOutSize;
    MKL_INT offsetBetween;
    MKL_INT secondSize;
    MKL_INT secondOutSize;
    MKL_INT offsetAfter;

    bool getPaddingFlag(MKL_INT fi, MKL_INT si)
    {
        return ((fi < 0) || (fi >= firstSize) || (si < 0) || (si >= secondSize));
    }

private:
    void swap(MKL_INT x, MKL_INT y)
    {
        MKL_INT tmp = x;
        x = y;
        y = tmp;
    }
};

}
}
}
}
}
}

#endif
