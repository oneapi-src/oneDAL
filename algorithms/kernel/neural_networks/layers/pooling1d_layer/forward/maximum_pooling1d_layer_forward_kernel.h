/* file: maximum_pooling1d_layer_forward_kernel.h */
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
//  Declaration of template function that calculate forward pooling layer results.
//--

#ifndef __MAXIMUM_POOLING1D_LAYER_FORWARD_KERNEL_H__
#define __MAXIMUM_POOLING1D_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/pooling1d/maximum_pooling1d_layer_forward.h"
#include "neural_networks/layers/pooling1d/maximum_pooling1d_layer_forward_types.h"
#include "kernel.h"
#include "tensor.h"

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
namespace maximum_pooling1d
{
namespace forward
{
namespace internal
{

/**
 *  \brief Kernel for forward pooling layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class PoolingKernel : public Kernel
{
public:
    /* Computes the results of forward batch normalization layer */
    services::Status compute(const Tensor &dataTensor, Tensor &valueTensor,
                 Tensor *selectedPosTensor, const maximum_pooling1d::Parameter &parameter);
};

} // internal
} // forward
} // maximum_pooling1d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
