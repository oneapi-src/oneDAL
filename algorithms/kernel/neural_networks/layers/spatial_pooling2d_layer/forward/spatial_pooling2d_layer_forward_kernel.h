/* file: spatial_pooling2d_layer_forward_kernel.h */
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
//  Declaration of template function that calculate forward pooling layer relults.
//--


#ifndef __SPATIAL_POOLING2D_LAYER_FORWARD_KERNEL_H__
#define __SPATIAL_POOLING2D_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/spatial_pooling2d/spatial_pooling2d_layer_forward_types.h"
#include "spatial_pooling2d_layer_internal_types.h"
#include "kernel.h"
#include "tensor.h"

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
namespace spatial_pooling2d
{
namespace forward
{
namespace internal
{

/**
 *  \brief Kernel for forward pooling layer results computation
 */
template<typename algorithmFPType, CpuType cpu>
class BasePoolingKernel : public Kernel
{
public:
    virtual services::Status compute(const Tensor &dataTensor, Tensor &valueTensor,
                                     Tensor *selectedPosTensor, const spatial_pooling2d::Parameter &parameter);
protected:
    virtual Status computePooling(
        const pooling2d::Parameter &poolingPar,
        const spatial_pooling2d::Parameter &spatialParameter,
        const Tensor &dataTensor,
        Tensor &valueTensor,
        Tensor *selectedPosTensor) = 0;
};

/**
 *  \brief Kernel for forward pooling layer results computation
 */
template<typename algorithmFPType, spatial_pooling2d::internal::Method method, CpuType cpu>
class PoolingKernel : public BasePoolingKernel<algorithmFPType, cpu>
{
protected:
    Status computePooling(
        const pooling2d::Parameter &poolingPar,
        const spatial_pooling2d::Parameter &spatialParameter,
        const Tensor &dataTensor,
        Tensor &valueTensor,
        Tensor *selectedPosTensor);
};

template<typename algorithmFPType, CpuType cpu>
class PoolingKernel<algorithmFPType, spatial_pooling2d::internal::maximum, cpu> : public BasePoolingKernel<algorithmFPType, cpu>
{
protected:
    Status computePooling(
        const pooling2d::Parameter &poolingPar,
        const spatial_pooling2d::Parameter &spatialParameter,
        const Tensor &dataTensor,
        Tensor &valueTensor,
        Tensor *selectedPosTensor);
};

template<typename algorithmFPType, CpuType cpu>
class PoolingKernel<algorithmFPType, spatial_pooling2d::internal::stochastic, cpu> : public BasePoolingKernel<algorithmFPType, cpu>
{
protected:
    Status computePooling(
        const pooling2d::Parameter &poolingPar,
        const spatial_pooling2d::Parameter &spatialParameter,
        const Tensor &dataTensor,
        Tensor &valueTensor,
        Tensor *selectedPosTensor);
};

template<typename algorithmFPType, CpuType cpu>
class PoolingKernel<algorithmFPType, spatial_pooling2d::internal::average, cpu> : public BasePoolingKernel<algorithmFPType, cpu>
{
private:
    Status computePooling(
        const pooling2d::Parameter &poolingPar,
        const spatial_pooling2d::Parameter &spatialParameter,
        const Tensor &dataTensor,
        Tensor &valueTensor,
        Tensor *selectedPosTensor);
};
} // internal
} // forward
} // spatial_pooling2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
