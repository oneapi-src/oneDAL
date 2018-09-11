/* file: softmax_cross_layer_backward_kernel.h */
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
//  Implementation of the backward softmax cross layer
//--


#ifndef __SOFTMAX_CROSS_LAYER_BACKWARD_KERNEL_H__
#define __SOFTMAX_CROSS_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/loss/softmax_cross_layer.h"
#include "neural_networks/layers/loss/softmax_cross_layer_types.h"
#include "neural_networks/layers/loss/softmax_cross_layer_backward_types.h"
#include "kernel.h"
#include "service_math.h"
#include "service_error_handling.h"
#include "service_tensor.h"
#include "numeric_table.h"
#include "threading.h"
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
namespace loss
{
namespace softmax_cross
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for softmax_cross calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class SoftmaxCrossKernel : public Kernel
{
public:
    Status compute(
        const Tensor &probTensor,
        const Tensor &groundTruthTensor,
        const softmax_cross::Parameter &parameter,
        Tensor &resultTensor);

private:
    const size_t _nRowsInBlock = 5000;
    Status processBlock(
        const Tensor &probTensor,
        const Tensor &groundTruthTensor,
        const size_t nProcessedRows,
        const size_t nRowsInCurrentBlock,
        const size_t dim,
        Tensor &gradientTensor);

};

} // namespace internal
} // namespace backward
} // namespace softmax_cross
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
