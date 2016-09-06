/* file: pooling1d_layer_impl.i */
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
//  Common classes for 1D pooling layers
//--
*/


#ifndef __POOLING1D_LAYER_IMPL_I__
#define __POOLING1D_LAYER_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace pooling1d
{
namespace internal
{

struct Parameter
{
    /*
     * Input data tensor of size dims is viewed by this method as a 3-dimensional tensor of size:
     * offsetBefore * firstSize * offsetAfter
     */
    Parameter(size_t index, size_t inputPadding, size_t inputStride, size_t inputKernelSize,
              Tensor *dataTensor, const Collection<size_t> &dims, const Collection<size_t> &valueDims) :
        padding(inputPadding), stride(inputStride), kernelSize(inputKernelSize)
    {
        MKL_INT nDims = (MKL_INT)dims.size();
        offsetBefore = (index == 0 ? 1 : dataTensor->getSize(0, index));
        firstSize = dims[index];
        firstOutSize = valueDims[index];
        offsetAfter = ((MKL_INT)index == nDims - 1 ? 1 : dataTensor->getSize(index + 1, nDims - (MKL_INT)index - 1));
    }

    MKL_INT padding;
    MKL_INT stride;
    MKL_INT kernelSize;

    MKL_INT offsetBefore;
    MKL_INT firstSize;
    MKL_INT firstOutSize;
    MKL_INT offsetAfter;
};

}
}
}
}
}
}

#endif
