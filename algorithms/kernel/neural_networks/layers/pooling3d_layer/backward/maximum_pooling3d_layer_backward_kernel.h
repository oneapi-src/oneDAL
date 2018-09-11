/* file: maximum_pooling3d_layer_backward_kernel.h */
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
//  Declaration of template function that calculate backward pooling layer relults.
//--


#ifndef __MAXIMUM_POOLING3D_LAYER_BACKWARD_KERNEL_H__
#define __MAXIMUM_POOLING3D_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/pooling3d/maximum_pooling3d_layer_backward.h"
#include "neural_networks/layers/pooling3d/maximum_pooling3d_layer_backward_types.h"
#include "kernel.h"
#include "tensor.h"
#include "service_blas.h"

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
namespace maximum_pooling3d
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for backward pooling layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class PoolingKernel : public Kernel
{
public:
    services::Status compute(const Tensor &inputGradTensor,
                const Tensor &selectedPosTensor, Tensor &gradTensor,
                const maximum_pooling3d::Parameter &parameter);

protected:
    void recurrentCompute(size_t d,
                DAAL_INT *ii, DAAL_INT *ik, DAAL_INT *iv,
                const DAAL_INT *padding, const DAAL_INT *stride, const DAAL_INT *kernelSize,
                const DAAL_INT* gradSize, const DAAL_INT* inputSize,
                const DAAL_INT* offset, DAAL_INT* gradOffset, DAAL_INT* inputOffset,
                const algorithmFPType *inputGrad, algorithmFPType *grad, const int *selectedPos);

    static size_t const nKernelDims = 3; /*!< Number of kernel dimensions */
};

} // internal
} // backward
} // maximum_pooling3d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
