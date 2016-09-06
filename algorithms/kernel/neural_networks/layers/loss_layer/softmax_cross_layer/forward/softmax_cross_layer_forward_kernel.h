/* file: softmax_cross_layer_forward_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
#include "numeric_table.h"
#include "threading.h"

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
    void compute(const softmax_cross::forward::Input *input, const softmax_cross::Parameter *parameter,
                 softmax_cross::forward::Result *result);

private:
    const size_t _nRowsInBlock = 5000;
    algorithmFPType _eps;

    inline algorithmFPType processBlock(SharedPtr<Tensor> inputTensor,
            SharedPtr<Tensor> groundTruthTensor,
            size_t nProcessedRows, size_t nRowsInCurrentBlock,
            SharedPtr<Tensor> probabilitiesTensor,
            Error *localError);

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
