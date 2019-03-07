/* file: transposed_conv2d_layer_backward_kernel.h */
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

//++
//  Declaration of template function that calculate transposed convolution 2d.
//--


#ifndef __TRANSPOSED_CONV2D_LAYER_BACKWARD_KERNEL_H__
#define __TRANSPOSED_CONV2D_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/transposed_conv2d/transposed_conv2d_layer.h"
#include "neural_networks/layers/transposed_conv2d/transposed_conv2d_layer_types.h"
#include "kernel.h"
#include "service_math.h"
#include "numeric_table.h"

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
namespace transposed_conv2d
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for transposed convolution 2d calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TransposedConv2dKernel : public Kernel
{
public:
    services::Status compute(const Tensor &inGradTensor, const Tensor &xTensor, const Tensor &wTensor,
        const transposed_conv2d::Parameter &parameter, Tensor &wDerTensor, Tensor &bDerTensor, Tensor &resultTensor);
};

} // internal
} // backward
} // transposed_conv2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
