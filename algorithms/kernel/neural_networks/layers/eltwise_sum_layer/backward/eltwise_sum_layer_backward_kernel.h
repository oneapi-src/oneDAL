/* file: eltwise_sum_layer_backward_kernel.h */
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
//  Declaration of template function that calculate eltwise_sums.
//--


#ifndef __ELTWISE_SUM_LAYER_BACKWARD_KERNEL_H__
#define __ELTWISE_SUM_LAYER_BACKWARD_KERNEL_H__

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
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for eltwise_sum calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class EltwiseSumKernel : public Kernel
{
public:
    services::Status compute(Tensor *inputGradient, Tensor *coefficients,
        Tensor **outputs, size_t nOutputs);

private:
    services::Status processOutputTensor(Tensor *inputGradient,
        const algorithmFPType *coefficientsArray, Tensor *output, size_t outputIndex);

    bool checkForInPlace(const Tensor *inputGradient, const Tensor *coefficients,
        Tensor **outputs, size_t nOutputs);
};

} // namespace internal
} // namespace backward
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
