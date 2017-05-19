/* file: prelu_layer_forward_kernel.h */
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
//  Implementation of prelu calculation functions.
//--


#ifndef __PRELU_LAYER_FORWARD_KERNEL_H__
#define __PRELU_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/prelu/prelu_layer.h"
#include "neural_networks/layers/prelu/prelu_layer_types.h"
#include "kernel.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
#include "threading.h"
#include "layers_threading.h"
#include "service_memory.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::internal;
using namespace daal::algorithms::neural_networks::layers::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace prelu
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for prelu calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class PReLUKernel : public Kernel
{
public:
    services::Status compute(Tensor *inputTensor, Tensor *weightsTensor, Tensor *resultTensor, const prelu::Parameter *parameter);
private:
    void getNumberOfFixedDimensions(TensorOffsetLayout &inputLayout, const Collection<size_t> &dims, size_t wEndDim, size_t &fDimN, size_t &wOffset, size_t minElementsNumInBlock);
    void processBlock(Tensor *inputTensor, Tensor *resultTensor, const algorithmFPType *wArray, size_t fDimN, size_t *fDims,
                      const TensorOffsetLayout &layout, size_t wSize, size_t wOffset, size_t wStart, size_t wLen, const Collection<size_t> &inDims, Collection<size_t> wOffsets);

    size_t _nElemsInBlock = 1000;
};
} // internal
} // forward
} // prelu
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
