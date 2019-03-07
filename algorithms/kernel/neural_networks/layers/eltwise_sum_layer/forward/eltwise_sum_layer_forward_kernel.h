/* file: eltwise_sum_layer_forward_kernel.h */
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
//  Declaration of template function that calculate element-wise sum.
//--


#ifndef __ELTWISE_SUM_LAYER_FORWARD_KERNEL_H__
#define __ELTWISE_SUM_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/eltwise_sum/eltwise_sum_layer.h"
#include "neural_networks/layers/eltwise_sum/eltwise_sum_layer_types.h"

#include "kernel.h"
#include "layers_threading.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"

using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::neural_networks::layers::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace eltwise_sum
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for element-wise sum calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class EltwiseSumKernel : public Kernel
{
public:
    services::Status compute(Tensor **inputs, Tensor *value, Tensor *coefficients,
        Tensor *auxCoefficients, NumericTable *numberOfCoefficients, size_t nInputs);

private:
    services::Status computeGeneric(Tensor **inputs, Tensor *value,
        const algorithmFPType *coefficients, size_t nInputs);

    services::Status makeResultForBackward(Tensor *coefficients, Tensor *auxCoefficients,
        NumericTable *numberOfCoefficients, size_t nInputs);
};
} // namespace internal
} // namespace forward

} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
