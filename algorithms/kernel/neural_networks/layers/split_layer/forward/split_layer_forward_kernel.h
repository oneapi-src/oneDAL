/* file: split_layer_forward_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Declaration of template function that calculate splits.
//--


#ifndef __SPLIT_LAYER_FORWARD_KERNEL_H__
#define __SPLIT_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/split/split_layer.h"
#include "neural_networks/layers/split/split_layer_types.h"
#include "kernel.h"
#include "service_dnn.h"
#include "service_dnn_internal.h"
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
namespace split
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for split calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class SplitKernel : public Kernel
{
public:
    services::Status compute(Tensor *inputTensor, Tensor *resultTensors[], size_t nOutputs);

private:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;

    const size_t _nRowsInBlock = 5000;

    inline Status processBlock(Tensor *inputTensor, size_t nProcessedRows, size_t nRowsInCurrentBlock, Tensor *resultTensor);
};
} // internal
} // forward
} // split
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
