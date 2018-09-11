/* file: prelu_layer_forward_kernel.h */
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
    services::Status compute(const Tensor &inputTensor, const Tensor &wTensor, Tensor &resultTensor, const prelu::Parameter &parameter);
private:
    void getNumberOfFixedDimensions(TensorOffsetLayout &inputLayout, const Collection<size_t> &dims, size_t wEndDim, size_t &fDimN, size_t &wOffset, size_t minElementsNumInBlock);
    Status processBlock(const Tensor &inputTensor, Tensor &resultTensor, const algorithmFPType *wArray, size_t fDimN,
                                                             size_t *fDims, const TensorOffsetLayout &layout, size_t wSize,
                                                             size_t wOffset, size_t wStart, size_t wLen, const Collection<size_t> &inDims,
                                                             const Collection<size_t> &wOffsets);

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
