/* file: spatial_maximum_pooling2d_layer_forward_batch_container.h */
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

/*
//++
//  Implementation of forward pooling layer container.
//--
*/

#ifndef __SPATIAL_PYRAMID_MAXIMUM_POOLING2D_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __SPATIAL_PYRAMID_MAXIMUM_POOLING2D_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/spatial_pooling2d/spatial_maximum_pooling2d_layer.h"
#include "spatial_pooling2d_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_maximum_pooling2d
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, defaultDense, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(spatial_pooling2d::forward::internal::PoolingKernel, algorithmFPType, spatial_pooling2d::internal::maximum);
}

template<typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, defaultDense, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, CpuType cpu>
services::Status BatchContainer<algorithmFPType, defaultDense, cpu>::compute()
{
    spatial_maximum_pooling2d::forward::Input *input = static_cast<spatial_maximum_pooling2d::forward::Input *>(_in);
    spatial_maximum_pooling2d::forward::Result *result = static_cast<spatial_maximum_pooling2d::forward::Result *>(_res);
    spatial_maximum_pooling2d::Parameter *parameter = static_cast<spatial_maximum_pooling2d::Parameter *>(_par);

    data_management::Tensor *dataTensor = input->get(layers::forward::data).get();
    data_management::Tensor *valueTensor = result->get(layers::forward::value).get();
    data_management::Tensor *selectedPosTensor = nullptr;
    if(parameter->predictionStage == false)
    {
        selectedPosTensor = result->get(auxSelectedIndices).get();
    }

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, spatial_pooling2d::forward::internal::PoolingKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, spatial_pooling2d::internal::maximum),   \
                       compute, *dataTensor, *valueTensor, selectedPosTensor, *parameter);
}
} // namespace interface1
} // namespace forward

} // namespace spatial_maximum_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
