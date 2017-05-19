/* file: maximum_pooling3d_layer_forward_kernel.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#ifndef __MAXIMUM_POOLING3D_LAYER_FORWARD_KERNEL_H__
#define __MAXIMUM_POOLING3D_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/pooling3d/maximum_pooling3d_layer_forward.h"
#include "neural_networks/layers/pooling3d/maximum_pooling3d_layer_forward_types.h"
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
    /* Computes the results of forward batch normalization layer */
    services::Status compute(Tensor *dataTensor, Tensor *valueTensor,
                 Tensor *selectedPosTensor, const maximum_pooling3d::Parameter *parameter);

protected:
    void recurrentCompute(size_t d,
                DAAL_INT *ii, DAAL_INT *ik, DAAL_INT *iv,
                const DAAL_INT *padding, const DAAL_INT *stride, const DAAL_INT *kernelSize,
                const DAAL_INT* dataSize, const DAAL_INT* valueSize,
                const DAAL_INT* offset, DAAL_INT* dataOffset, DAAL_INT* valueOffset,
                const algorithmFPType *data, algorithmFPType *value, int *selectedPos);

    static size_t const nKernelDims = 3; /*!< Number of kernel dimensions */
};

} // internal
} // forward
} // maximum_pooling3d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
