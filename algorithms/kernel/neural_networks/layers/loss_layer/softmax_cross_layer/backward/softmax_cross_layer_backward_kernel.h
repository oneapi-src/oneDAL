/* file: softmax_cross_layer_backward_kernel.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the backward softmax cross layer
//--


#ifndef __SOFTMAX_CROSS_LAYER_BACKWARD_KERNEL_H__
#define __SOFTMAX_CROSS_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/loss/softmax_cross_layer.h"
#include "neural_networks/layers/loss/softmax_cross_layer_types.h"
#include "neural_networks/layers/loss/softmax_cross_layer_backward_types.h"
#include "kernel.h"
#include "service_math.h"
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
    services::Status compute(Tensor *probTensor, Tensor *groundTruthTensor, const softmax_cross::Parameter *parameter, Tensor *resultTensor);

private:
    const size_t _nRowsInBlock = 5000;
    size_t _dim;
    void processBlock(Tensor *probTensor,
                      Tensor *groundTruthTensor,
                      size_t nProcessedRows, size_t nRowsInCurrentBlock,
                      Tensor *gradientTensor,
                      Error *localError);

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
