/* file: service_tensor.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

#include "service_tensor.h"

namespace daal
{
namespace internal
{

size_t computeTensorDimensionsProd(const Tensor *tensor, size_t axisFrom, size_t axisTo)
{
    if (!tensor || !tensor->getNumberOfDimensions()) { return 0; }
    const services::Collection<size_t> &dimensions = tensor->getDimensions();

    size_t offset = 1;
    for (size_t i = axisFrom; i < axisTo; i++)
    {
        offset *= dimensions[i];
    }
    return offset;
}

size_t computeTensorOffsetBeforeAxis(const Tensor *tensor, size_t axis)
{
    return computeTensorDimensionsProd(tensor, 0, axis);
}

size_t computeTensorOffsetAfterAxis(const Tensor *tensor, size_t axis)
{
    if (!tensor) { return 0; }
    return computeTensorDimensionsProd(tensor, axis + 1, tensor->getNumberOfDimensions());
}

} // internal namespace
} // daal namespace
