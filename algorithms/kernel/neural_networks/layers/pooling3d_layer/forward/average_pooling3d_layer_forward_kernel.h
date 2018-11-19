/* file: average_pooling3d_layer_forward_kernel.h */
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
//  Declaration of template function that calculate forward pooling layer results.
//--

#ifndef __AVERAGE_POOLING3D_LAYER_FORWARD_KERNEL_H__
#define __AVERAGE_POOLING3D_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/pooling3d/average_pooling3d_layer_forward.h"
#include "neural_networks/layers/pooling3d/average_pooling3d_layer_forward_types.h"
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
namespace average_pooling3d
{
namespace forward
{
namespace internal
{

/**
 *  \brief Kernel for forward pooling layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class PoolingKernel : public Kernel
{
public:
    /* Computes the results of forward pooling layer */
    services::Status compute(const Tensor &dataTensor, const average_pooling3d::Parameter &parameter,Tensor &valueTensor);

protected:
    void recurrentCompute(size_t d,
                DAAL_INT *ii, DAAL_INT *ik, DAAL_INT *iv,
                const DAAL_INT *padding, const DAAL_INT *stride, const DAAL_INT *kernelSize,
                const DAAL_INT* dataSize, const DAAL_INT* valueSize,
                const DAAL_INT* offset, DAAL_INT* dataOffset, DAAL_INT* valueOffset,
                const algorithmFPType *data, algorithmFPType *value, algorithmFPType divisor);

    static size_t const nKernelDims = 3; /*!< Number of kernel dimensions */
};

} // internal
} // forward
} // average_pooling3d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
