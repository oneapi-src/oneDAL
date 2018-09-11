/* file: smoothrelu_layer_backward_kernel.h */
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
// Implementation of the backward smooth rectifier linear unit (smooth relu) layer
//--


#ifndef __SMOOTHRELU_LAYER_BACKWARD_KERNEL_H__
#define __SMOOTHRELU_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/smoothrelu/smoothrelu_layer.h"
#include "neural_networks/layers/smoothrelu/smoothrelu_layer_types.h"
#include "kernel.h"
#include "service_math.h"
#include "numeric_table.h"
#include "service_blas.h"
#include "layers_threading.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::neural_networks::layers::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace smoothrelu
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for smoothrelu calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class SmoothReLUKernel : public Kernel
{
public:
    services::Status compute(const Tensor &inputGradientTensor, const Tensor &forwardDataTensor, Tensor &resultTensor);
};

} // internal
} // backward
} // smoothrelu
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
