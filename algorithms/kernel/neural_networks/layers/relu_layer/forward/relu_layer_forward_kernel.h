/* file: relu_layer_forward_kernel.h */
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
//  Declaration of template function that calculate relus.
//--


#ifndef __RELU_LAYER_FORWARD_KERNEL_H__
#define __RELU_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/relu/relu_layer.h"
#include "neural_networks/layers/relu/relu_layer_types.h"
#include "kernel.h"
#include "layers_threading.h"
#include "service_dnn.h"
#include "service_dnn_internal.h"

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
namespace relu
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for relu calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class ReLUKernel : public Kernel
{
public:
    services::Status compute(const Tensor &inputTensor, Tensor &resultTensor);

    ~ReLUKernel()
    {
        if (reluPrim)
        {
            dnn::xDelete(reluPrim);
        }
    }

private:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;

    dnnPrimitive_t reluPrim = NULL;
};

} // internal
} // forward
} // relu
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
