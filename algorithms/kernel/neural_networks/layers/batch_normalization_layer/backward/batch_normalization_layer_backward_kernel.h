/* file: batch_normalization_layer_backward_kernel.h */
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
//  Declaration of template function that calculate backward batch normalization layer relults.
//--


#ifndef __BATCH_NORMALIZATION_LAYER_BACKWARD_KERNEL_H__
#define __BATCH_NORMALIZATION_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/batch_normalization/batch_normalization_layer_backward.h"
#include "neural_networks/layers/batch_normalization/batch_normalization_layer_backward_types.h"
#include "kernel.h"
#include "tensor.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for backward batch normalization layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class BatchNormalizationKernel : public Kernel
{
public:
    services::Status compute(Tensor &gradientTensor,
                             const Tensor &weightsTensor,
                             const Tensor &stDevTensor,
                             const Tensor &inputGradientTensor,
                             const Tensor &dataTensor,
                             const Tensor &meanTensor,
                             Tensor &weightsDerTensor,
                             Tensor &biasesDerTensor,
                             const batch_normalization::Parameter &parameter);

};

} // internal
} // backward
} // batch_normalization
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
