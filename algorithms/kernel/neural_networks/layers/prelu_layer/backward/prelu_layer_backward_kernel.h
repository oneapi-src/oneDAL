/* file: prelu_layer_backward_kernel.h */
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
    PReLUTask(const Tensor &_inGradTensor, const Tensor &_xTensor, const Tensor &_wTensor, Tensor &_wDerTensor, Tensor &_resultTensor, const prelu::Parameter &parameter);

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

    const Tensor &inGradTensor;
    const Tensor &xTensor;
    const Tensor &wTensor;
    Tensor &wDerTensor;
    Tensor &resultTensor;

    algorithmFPType invN;

    services::Status status;

};

/**
 *  \brief Kernel for prelu calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class PReLUKernel : public Kernel
{
public:
    services::Status compute( PReLUTask<algorithmFPType, method, cpu> &task,
                  const prelu::Parameter &parameter );

protected:
    services::Status computeGradientBlock( PReLUTask<algorithmFPType, method, cpu> &task,
                               size_t *fDims,
                               algorithmFPType *wDerArray );

    services::Status computeDerivativesBlock( PReLUTask<algorithmFPType, method, cpu> &task,
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
