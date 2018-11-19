/* file: split_layer_forward_kernel.h */
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
