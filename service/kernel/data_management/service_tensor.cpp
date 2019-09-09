/* file: service_tensor.cpp */
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
