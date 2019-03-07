/* file: stochastic_pooling2d_layer_forward_kernel.h */
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
//  Declaration of template function that calculate forward pooling layer results.
//--

#ifndef __STOCHASTIC_POOLING2D_LAYER_FORWARD_KERNEL_H__
#define __STOCHASTIC_POOLING2D_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/pooling2d/stochastic_pooling2d_layer_forward.h"
#include "neural_networks/layers/pooling2d/stochastic_pooling2d_layer_forward_types.h"
#include "pooling2d_layer_internal_parameter.h"
#include "service_blas.h"
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
    services::Status compute(const Tensor &dataTensor, Tensor &valueTensor,
                 Tensor *selectedPosTensor, const stochastic_pooling2d::Parameter &parameter, engines::BatchBase &engine);

private:
    algorithmFPType invIntMaxVal;

    inline void computeWeightedAverage(
    const algorithmFPType *dataSlice,
    DAAL_INT f,
    DAAL_INT s,
    algorithmFPType *kernelWeights,
    pooling2d::internal::Parameter &par,
    algorithmFPType &value);

    inline void getMultivariateRandomDataValue(
    const algorithmFPType *dataSlice,
    DAAL_INT f,
    DAAL_INT s,
    algorithmFPType *weights,
    size_t nWeights,
    pooling2d::internal::Parameter &par,
    algorithmFPType &value,
    int &selectedPos);

    inline size_t getMultinomialRandomValue(algorithmFPType *weights, size_t nWeights, const int uniformRandVal);
    Status getUniformRandFrom0to1(int* uniformRand, const size_t nUniformRand, engines::BatchBase &engine);
};

} // internal
} // forward
} // stochastic_pooling2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
