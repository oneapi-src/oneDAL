/* file: dropout_layer_backward_kernel.h */
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
//  Implementation of the backward dropout layer
//--


#ifndef __DROPOUT_LAYER_BACKWARD_KERNEL_H__
#define __DROPOUT_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/dropout/dropout_layer.h"
#include "neural_networks/layers/dropout/dropout_layer_types.h"
#include "kernel.h"
#include "service_math.h"
#include "numeric_table.h"
#include "service_tensor.h"

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
namespace backward
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
    services::Status compute(const Tensor &inputGradientTable,
                             const Tensor &maskTable,
                             Tensor &resultTable);

private:
    const size_t _nRowsInBlock = 5000;

    inline services::Status processBlock(
        const Tensor &inputGradientTable,
        const Tensor &maskTable,
        const size_t nProcessedRows,
        const size_t nRowsInCurrentBlock,
        Tensor &resultTable);
};

} // internal
} // backward
} // dropout
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
