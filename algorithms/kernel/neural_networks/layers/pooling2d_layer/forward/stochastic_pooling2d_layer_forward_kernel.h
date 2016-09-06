/* file: stochastic_pooling2d_layer_forward_kernel.h */
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

#ifndef __STOCHASTIC_POOLING2D_LAYER_FORWARD_KERNEL_H__
#define __STOCHASTIC_POOLING2D_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/pooling2d/stochastic_pooling2d_layer_forward.h"
#include "neural_networks/layers/pooling2d/stochastic_pooling2d_layer_forward_types.h"
#include "pooling2d_layer_internal_parameter.h"
#include "service_blas.h"
#include "service_rng.h"
#include "kernel.h"
#include "tensor.h"
#include "service_memory.h"
#include "service_data_utils.h"
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
namespace stochastic_pooling2d
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
    void compute(Tensor *dataTensor, Tensor *valueTensor,
                 Tensor *selectedPosTensor, const stochastic_pooling2d::Parameter *parameter);

private:
    algorithmFPType invIntMaxVal;

    inline void computeWeightedAverage(
    const algorithmFPType *dataSlice,
    MKL_INT f,
    MKL_INT s,
    algorithmFPType *kernelWeights,
    pooling2d::internal::Parameter &par,
    algorithmFPType &value);

    inline void getMultivariateRandomDataValue(
    const algorithmFPType *dataSlice,
    MKL_INT f,
    MKL_INT s,
    algorithmFPType *weights,
    size_t nWeights,
    pooling2d::internal::Parameter &par,
    algorithmFPType &value,
    int &selectedPos);

    inline size_t getMultinomialRandomValue(algorithmFPType *weights, size_t nWeights, const int uniformRandVal);
    void getUniformRandFrom0to1(int* uniformRand, size_t nUniformRand, size_t seed);
};

} // internal
} // forward
} // stochastic_pooling2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
