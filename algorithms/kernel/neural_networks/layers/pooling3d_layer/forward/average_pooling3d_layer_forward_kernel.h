/* file: average_pooling3d_layer_forward_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
    void compute(const average_pooling3d::forward::Input *input,
                 const average_pooling3d::Parameter *parameter,
                 average_pooling3d::forward::Result *result);

protected:
    void recurrentCompute(size_t d,
                MKL_INT *ii, MKL_INT *ik, MKL_INT *iv,
                const MKL_INT *padding, const MKL_INT *stride, const MKL_INT *kernelSize,
                const MKL_INT* dataSize, const MKL_INT* valueSize,
                const MKL_INT* offset, MKL_INT* dataOffset, MKL_INT* valueOffset,
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
