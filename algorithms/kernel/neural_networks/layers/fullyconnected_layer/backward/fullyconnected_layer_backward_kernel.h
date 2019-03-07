/* file: fullyconnected_layer_backward_kernel.h */
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
//  Declaration of template function that calculate fullyconnecteds.
//--


#ifndef __FULLYCONNECTED_LAYER_BACKWARD_KERNEL_H__
#define __FULLYCONNECTED_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/fullyconnected/fullyconnected_layer.h"
#include "neural_networks/layers/fullyconnected/fullyconnected_layer_types.h"
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
namespace fullyconnected
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for fullyconnected calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class FullyconnectedKernel : public Kernel
{
public:
    services::Status compute(const Tensor &inGradTensor, const Tensor &xTensor, const Tensor &wTensor, Tensor &wDerTensor,
                                                                 Tensor &bDerTensor, Tensor &resultTensor, const fullyconnected::Parameter &parameter);
};

} // internal
} // backward
} // fullyconnected
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
