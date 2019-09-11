/* file: pooling1d_layer_impl.i */
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
              const Tensor &dataTensor, const Collection<size_t> &dims, const Collection<size_t> &valueDims) :
        padding(inputPadding), stride(inputStride), kernelSize(inputKernelSize)
    {
        DAAL_INT nDims = (DAAL_INT)dims.size();
        offsetBefore = (index == 0 ? 1 : dataTensor.getSize(0, index));
        firstSize = dims[index];
        firstOutSize = valueDims[index];
        offsetAfter = ((DAAL_INT)index == nDims - 1 ? 1 : dataTensor.getSize(index + 1, nDims - (DAAL_INT)index - 1));
    }

    DAAL_INT padding;
    DAAL_INT stride;
    DAAL_INT kernelSize;

    DAAL_INT offsetBefore;
    DAAL_INT firstSize;
    DAAL_INT firstOutSize;
    DAAL_INT offsetAfter;
};

}
}
}
}
}
}

#endif
