/* file: softmax_cross_layer_forward_kernel.h */
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
//  Implementation of the forward softmax cross layer
//--


#ifndef __SOFTMAX_CROSS_LAYER_FORWARD_KERNEL_H__
#define __SOFTMAX_CROSS_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/softmax_layer/forward/softmax_layer_forward_kernel.h"
#include "neural_networks/layers/loss/softmax_cross_layer.h"
#include "neural_networks/layers/loss/softmax_cross_layer_types.h"
#include "neural_networks/layers/loss/softmax_cross_layer_forward_types.h"
#include "kernel.h"
#include "service_rng.h"
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
namespace forward
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
    services::Status compute(
        const Tensor &inputTensor,
        const Tensor &groundTruthTensor,
        const softmax_cross::Parameter &parameter,
        Tensor &probabilitiesTensor,
        Tensor &resultTensor);

private:
    const size_t _nRowsInBlock = 5000;

    inline Status processBlock(
        const Tensor &inputTensor,
        const Tensor &groundTruthTensor,
        const size_t nProcessedRows,
        const size_t nRowsInCurrentBlock,
        Tensor &probabilitiesTensor,
        const size_t dim,
        const algorithmFPType eps,
        SafeStatus &safeStat,
        algorithmFPType &partialLoss);
};

} // internal
} // forward
} // softmax_cross
} // loss
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
