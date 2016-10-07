/* file: spatial_pooling2d_layer_backward_kernel.h */
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
//  Declaration of template function that calculate backward pooling layer relults.
//--


#ifndef __SPATIAL_POOLING2D_LAYER_BACKWARD_KERNEL_H__
#define __SPATIAL_POOLING2D_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/spatial_pooling2d/spatial_pooling2d_layer_backward_types.h"
#include "spatial_pooling2d_layer_internal_types.h"
#include "kernel.h"
#include "tensor.h"
#include "spatial_pooling2d_layer_backward_task.h"

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
namespace backward
{
namespace internal
{

using namespace spatial_pooling2d::internal;

/**
 *  \brief Kernel for forward pooling layer results computation
 */
template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT BasePoolingKernel : public Kernel
{
public:
    virtual void compute(Tensor *inputGradientTensor,
                 Tensor *gradientTensor,
                 NumericTable *inputDimensions,
                 Tensor *selectedPosTensor,
                 const spatial_pooling2d::Parameter *parameter);

protected:
    virtual BasePoolingTask<algorithmFPType, cpu> *createTask(Tensor *_inputGradientTensor,
                                                              Tensor *_gradientTensor,
                                                              NumericTable *_inputDimensions,
                                                              Tensor *_selectedPosTensor,
                                                              const spatial_pooling2d::Parameter *) = 0;

    virtual void computePooling(BasePoolingTask<algorithmFPType, cpu> *task) = 0;

    void mergeToResult(Tensor *gradient,
                       Tensor *partialGradient,
                       size_t slice,
                       const TensorOffsetLayout &targetOutLayout);
};

/**
 *  \brief Kernel for forward pooling layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT PoolingKernel : public BasePoolingKernel<algorithmFPType, cpu>
{
protected:
    void computePooling(BasePoolingTask<algorithmFPType, cpu> *task) DAAL_C11_OVERRIDE;
};

template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT PoolingKernel<algorithmFPType, maximum, cpu> : public BasePoolingKernel<algorithmFPType, cpu>
{
protected:
    PoolingTask<algorithmFPType, maximum, cpu> *createTask(Tensor *_inputGradientTensor,
                                                           Tensor *_gradientTensor,
                                                           NumericTable *_inputDimensions,
                                                           Tensor *_selectedPosTensor,
                                                           const spatial_pooling2d::Parameter *parameter) DAAL_C11_OVERRIDE
    {
        return new PoolingTask<algorithmFPType, maximum, cpu>(_inputGradientTensor, _gradientTensor, _inputDimensions, _selectedPosTensor, parameter);
    }
    void computePooling(BasePoolingTask<algorithmFPType, cpu> *task) DAAL_C11_OVERRIDE;
};

template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT PoolingKernel<algorithmFPType, stochastic, cpu> : public BasePoolingKernel<algorithmFPType, cpu>
{
protected:
    PoolingTask<algorithmFPType, stochastic, cpu> *createTask(Tensor *_inputGradientTensor,
                                                              Tensor *_gradientTensor,
                                                              NumericTable *_inputDimensions,
                                                              Tensor *_selectedPosTensor,
                                                              const spatial_pooling2d::Parameter *parameter) DAAL_C11_OVERRIDE
    {
        return new PoolingTask<algorithmFPType, stochastic, cpu>(_inputGradientTensor, _gradientTensor, _inputDimensions, _selectedPosTensor, parameter);
    }
    void computePooling(BasePoolingTask<algorithmFPType, cpu> *task) DAAL_C11_OVERRIDE;
};

template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT PoolingKernel<algorithmFPType, average, cpu> : public BasePoolingKernel<algorithmFPType, cpu>
{
protected:
    PoolingTask<algorithmFPType, average, cpu> *createTask(Tensor *_inputGradientTensor,
                                                           Tensor *_gradientTensor,
                                                           NumericTable *_inputDimensions,
                                                           Tensor *_selectedPosTensor,
                                                           const spatial_pooling2d::Parameter *parameter) DAAL_C11_OVERRIDE
    {
        return new PoolingTask<algorithmFPType, average, cpu>(_inputGradientTensor, _gradientTensor, _inputDimensions, parameter);
    }

    void computePooling(BasePoolingTask<algorithmFPType, cpu> *task) DAAL_C11_OVERRIDE;
};

} // internal
} // backward
} // spatial_pooling2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
