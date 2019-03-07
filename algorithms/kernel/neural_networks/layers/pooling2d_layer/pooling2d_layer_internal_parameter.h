/* file: pooling2d_layer_internal_parameter.h */
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
              const Tensor &dataTensor, const Collection<size_t> &dims, const Collection<size_t> &valueDims);

    DAAL_INT firstIndex;
    DAAL_INT secondIndex;
    DAAL_INT firstPadding;
    DAAL_INT secondPadding;
    DAAL_INT firstStride;
    DAAL_INT secondStride;
    DAAL_INT firstKernelSize;
    DAAL_INT secondKernelSize;

    DAAL_INT offsetBefore;
    DAAL_INT firstSize;
    DAAL_INT firstOutSize;
    DAAL_INT offsetBetween;
    DAAL_INT secondSize;
    DAAL_INT secondOutSize;
    DAAL_INT offsetAfter;

    bool getPaddingFlag(DAAL_INT fi, DAAL_INT si)
    {
        return ((fi < 0) || (fi >= firstSize) || (si < 0) || (si >= secondSize));
    }

private:
    void swap(DAAL_INT &x, DAAL_INT &y)
    {
        DAAL_INT tmp = x;
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
