/* file: pooling2d_layer_internal_parameter.cpp */
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

#include "pooling2d_layer_internal_parameter.h"

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

Parameter::Parameter(const size_t *indices, const size_t *padding, const size_t *stride, const size_t *kernelSize,
              const Tensor &dataTensor, const Collection<size_t> &dims, const Collection<size_t> &valueDims) :
        firstIndex(indices[0]), secondIndex(indices[1]), firstPadding(padding[0]), secondPadding(padding[1]),
        firstStride(stride[0]), secondStride(stride[1]), firstKernelSize(kernelSize[0]), secondKernelSize(kernelSize[1])
    {
        if (firstIndex > secondIndex)
        {
            swap(firstIndex,   secondIndex);
            swap(firstPadding, secondPadding);
            swap(firstStride,  secondStride);
            swap(firstKernelSize, secondKernelSize);
        }

        size_t nDims = dims.size();
        offsetBefore = (firstIndex == 0 ? 1 : dataTensor.getSize(0, firstIndex));
        firstSize = dims[firstIndex];
        firstOutSize = valueDims[firstIndex];
        offsetBetween = (firstIndex + 1 == secondIndex ? 1 : dataTensor.getSize(firstIndex + 1, secondIndex - firstIndex - 1));
        secondSize = dims[secondIndex];
        secondOutSize = valueDims[secondIndex];
        offsetAfter = (secondIndex == nDims - 1 ? 1 : dataTensor.getSize(secondIndex + 1, nDims - secondIndex - 1));
    }

}
}
}
}
}
}
