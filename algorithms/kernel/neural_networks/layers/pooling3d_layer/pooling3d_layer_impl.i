/* file: pooling3d_layer_impl.i */
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
//  Common classes for 3D pooling layers
//--
*/

#ifndef __POOLING3D_LAYER_IMPL_I__
#define __POOLING3D_LAYER_IMPL_I__

#include "service_sort.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace pooling3d
{
namespace internal
{

template<CpuType cpu>
struct Parameter
{
    /*
     * Input data tensor is viewed by this method as a 7-dimensional tensor of size:
     * offset[0] * dataSize[0] * offset[1] * dataSize[1] * offset[2] * dataSize[2] * offset[3]
     */
    Parameter(const size_t *inputIndices, const size_t *inputPadding, const size_t *inputStride,
              const size_t *inputKernelSize,
              Tensor *dataTensor, const Collection<size_t> &dims, const Collection<size_t> &valueDims)
    {
        /* Get indices from the parameters of the layer */
        size_t indicesOrder[nKernelDims];
        for (size_t i = 0; i < nKernelDims; i++)
        {
            indices[i] = (MKL_INT)inputIndices[i];
            indicesOrder[i] = i;
        }

        /* Arrange indices in the ascending order and re-order other parameters respectively */
        daal::algorithms::internal::qSort<MKL_INT, size_t, cpu>(nKernelDims, indices, indicesOrder);

        for (size_t i = 0; i < nKernelDims; i++)
        {
            padding[i]    = (MKL_INT)inputPadding[indicesOrder[i]];
            stride[i]     = (MKL_INT)inputStride[indicesOrder[i]];
            kernelSize[i] = (MKL_INT)inputKernelSize[indicesOrder[i]];
        }

        /*
         * Input data tensor is viewed by this method as a 7-dimensional tensor of size:
         * offset[0] * dataSize[0] * offset[1] * dataSize[1] * offset[2] * dataSize[2] * offset[3]
         */
        for (size_t d = 0; d < nKernelDims; d++)
        {
            dataSize[d]  = (MKL_INT)dims[indices[d]];
            valueSize[d] = (MKL_INT)valueDims[indices[d]];
        }

        size_t nDims = dims.size();
        offset[0] = (indices[0] == 0 ? 1 : dataTensor->getSize(0, indices[0]));
        for (size_t d = 1; d < nKernelDims; d++)
        {
            offset[d] = (indices[d - 1] + 1 == indices[d] ? 1 : dataTensor->getSize(indices[d - 1] + 1, indices[d] - indices[d - 1] - 1));
        }
        offset[nKernelDims] = (indices[nKernelDims - 1] == nDims - 1 ? 1 :
            dataTensor->getSize(indices[nKernelDims - 1] + 1, nDims - indices[nKernelDims - 1] - 1));
    }

    static size_t const nKernelDims = 3; /*!< Number of kernel dimensions */

    MKL_INT indices[nKernelDims];
    MKL_INT padding[nKernelDims];
    MKL_INT stride[nKernelDims];
    MKL_INT kernelSize[nKernelDims];

    MKL_INT offset[nKernelDims + 1];
    MKL_INT dataSize[nKernelDims];      // size of the input 3D subtensor over which the kernel is applied
    MKL_INT valueSize[nKernelDims];     // size of the output 3D subtensor over which the kernel is applied
};

}
}
}
}
}
}

#endif
