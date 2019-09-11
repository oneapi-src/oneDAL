/* file: dropout_layer_forward_kernel.h */
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
//  Implementation of the forward dropout layer
//--


#ifndef __DROPOUT_LAYER_FORWARD_KERNEL_H__
#define __DROPOUT_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/dropout/dropout_layer.h"
#include "neural_networks/layers/dropout/dropout_layer_types.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_math.h"
#include "service_numeric_table.h"
#include "bernoulli_kernel.h"

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
namespace dropout
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for dropout calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DropoutKernel : public Kernel
{
public:
    DropoutKernel() : _retainRatio(0.5) {}
    services::Status compute(
        const Tensor &inputTensor,
        Tensor &resultTensor,
        Tensor *maskTensor,
        const dropout::Parameter &parameter);

    services::Status initialize(const dropout::Parameter &parameter);

    services::Status reset();

private:
    static const size_t _nRowsInBlock = 5000;

    algorithmFPType _retainRatio;
    engines::BatchBase *_engine;

    inline Status processBlock(
        const Tensor &inputTensor,
        const size_t nProcessedRows,
        const size_t nRowsInCurrentBlock,
        Tensor &resultTensor,
        Tensor *maskTensor,
        int *rngBuffer,
        const algorithmFPType inverseRetainRatio);

    inline Status processBlockPrediction(
        const Tensor &inputTensor,
        const size_t nProcessedRows,
        const size_t nRowsInCurrentBlock,
        Tensor &resultTensor);
};
} // internal
} // forward

} // dropout
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
