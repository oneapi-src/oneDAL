/* file: prelu_layer_backward_kernel.h */
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


#ifndef __PRELU_LAYER_BACKWARD_KERNEL_H__
#define __PRELU_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/prelu/prelu_layer.h"
#include "neural_networks/layers/prelu/prelu_layer_types.h"
#include "kernel.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
#include "service_memory.h"
#include "threading.h"
#include "layers_threading.h"

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
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
struct PReLUTask
{
    PReLUTask(Tensor *inGradTensor, Tensor *xTensor, Tensor *wTensor, Tensor *wDerTensor, Tensor *resultTensor, const prelu::Parameter *parameter);

    virtual ~PReLUTask() {};

    WriteOnlySubtensor<algorithmFPType, cpu> wDerBlock;
    ReadSubtensor<algorithmFPType, cpu> wBlock;

    const algorithmFPType *wArray;
    algorithmFPType *wDerArray;

    TensorOffsetLayout inputLayout;

    Collection<size_t> xDims;
    Collection<size_t> wOffsets;

    size_t wStart;
    size_t wLen;
    size_t wSize;
    size_t fDimN;
    size_t wOffset;
    size_t _nElemsInBlock = 1000;

    Tensor *inGradTensor;
    Tensor *xTensor;
    Tensor *wTensor;
    Tensor *wDerTensor;
    Tensor *resultTensor;

    algorithmFPType invN;

};

/**
 *  \brief Kernel for prelu calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class PReLUKernel : public Kernel
{
public:
    services::Status compute( PReLUTask<algorithmFPType, method, cpu> &task,
                  const prelu::Parameter *parameter );

protected:
    void computeGradientBlock( PReLUTask<algorithmFPType, method, cpu> &task,
                               size_t *fDims,
                               algorithmFPType *wDerArray );

    void computeDerivativesBlock( PReLUTask<algorithmFPType, method, cpu> &task,
                                  size_t *fDims,
                                  algorithmFPType *wDerArray );
};

} // internal
} // backward
} // prelu
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
