/* file: convolution2d_layer_forward_kernel.h */
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

//++
//  Declaration of template function that calculate convolution2ds.
//--


#ifndef __CONVOLUTION2D_LAYER_FORWARD_KERNEL_H__
#define __CONVOLUTION2D_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/convolution2d/convolution2d_layer.h"
#include "neural_networks/layers/convolution2d/convolution2d_layer_types.h"
#include "kernel.h"
#include "service_math.h"
#include "numeric_table.h"
#include "service_dnn.h"
#include "service_dnn_internal.h"
#include "layers_threading.h"

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
namespace convolution2d
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for convolution2d calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class Convolution2dKernel : public Kernel
{
public:
    Convolution2dKernel() : convPrim(NULL) {}

    services::Status compute(Tensor *inputTensor, Tensor *wTensor, Tensor *bTensor,
                                                                const convolution2d::Parameter &parameter, Tensor *resultTensor);

    services::Status initialize(const services::Collection<size_t>& inDimsFull, const services::Collection<size_t>& wDims,
                                                                const convolution2d::Parameter &parameter, const services::Collection<size_t> &outDimsFull);

    services::Status reset();

private:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;
    typedef daal::internal::DnnLayout<algorithmFPType, cpu> xDnnLayout;
    typedef daal::internal::DnnBuffer<algorithmFPType, cpu> xDnnBuffer;

    static const size_t dimension = 4;

    size_t inputSize    [ dimension ];
    size_t inputStrides [ dimension ];
    size_t outputSize   [ dimension ];
    size_t outputStrides[ dimension ];
    size_t filterSize   [ dimension + 1 ];
    size_t filterStrides[ dimension + 1 ];

    size_t  biasSize   [1];
    size_t  biasStrides[1];

    size_t convolutionStride[2];
    int    inputOffset      [2];

    xDnnLayout ltUserInput ;
    xDnnLayout ltUserFilt  ;
    xDnnLayout ltUserBias  ;
    xDnnLayout ltUserOutput;

    dnnPrimitive_t convPrim;
};
} // internal
} // forward

} // convolution2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
