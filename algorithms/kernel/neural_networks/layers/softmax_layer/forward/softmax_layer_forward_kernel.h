/* file: softmax_layer_forward_kernel.h */
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
//  Implementation of softmax layer.
//--


#ifndef __SOFTMAX_LAYER_FORWARD_KERNEL_H__
#define __SOFTMAX_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/softmax/softmax_layer.h"
#include "neural_networks/layers/softmax/softmax_layer_types.h"
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
namespace softmax
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for softmax calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class SoftmaxKernel : public Kernel
{
public:
    services::Status compute(const Tensor &inputTensor, const softmax::Parameter &parameter, Tensor &resultTensor);

private:
    const size_t _nRowsInBlock = 5000;
};
} // internal
} // forward

} // softmax
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
