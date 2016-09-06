/* file: batch_normalization_layer_forward_kernel.h */
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
//  Declaration of template function that calculate forward batch normalization layer relults.
//--

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_KERNEL_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/batch_normalization/batch_normalization_layer_forward.h"
#include "neural_networks/layers/batch_normalization/batch_normalization_layer_forward_types.h"
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
namespace batch_normalization
{
namespace forward
{
namespace internal
{
/**
 * \brief Structure for storing data used in itermediate computations
 *        in forward batch normalization layer
 */
template<typename algorithmFPType, Method method, CpuType cpu>
struct BatchNormalizationTask
{
    /*
     * Constructs the structure for storing data used in itermediate computations
     * in forward batch normalization layer
     */
    BatchNormalizationTask(const batch_normalization::forward::Input *input,
            batch_normalization::forward::Result *result, size_t dimension);

    virtual ~BatchNormalizationTask();

    /**
     * p-dimensional tensor of size n_1 * n_2 * ... * n_p
     * that stores forward batch normalization input data
     */
    SharedPtr<Tensor> inputTensor;
    SubtensorDescriptor<algorithmFPType> inputBlock;    /*!< Object that manages read operations required by input data tensor */
    algorithmFPType *data;                              /*!< Buffer that contains a block of values from input data tensor */

    /*
     * Input data tensor is viewed by the algorithm as a 3-dimensional tensor of size offsetBefore * dimensionSize * offsetAfter
     */
    size_t offsetBefore;    /*!< n_1 * ... * n_(k-1), where k is the dimension over which the normalization is performed */
    size_t dimensionSize;   /*!< n_k - size of the dimension over which the normalization is performed */
    size_t offsetAfter;     /*!< n_(k+1) * ... * n_p */

    SharedPtr<Tensor> meanTensor;                    /*!< 1-dimensional tensor of size n_k that stores mini-batch mean */
    SubtensorDescriptor<algorithmFPType> meanBlock;  /*!< Object that manages write operations required by mini-batch mean tensor */
    algorithmFPType *mean;                           /*!< Buffer that contains a block of values from mini-batch mean tensor */

    SharedPtr<Tensor> stDevTensor;                   /*!< 1-dimensional tensor of size n_k that stores mini-batch standard deviation */
    SubtensorDescriptor<algorithmFPType> stDevBlock; /*!< Object that manages write operations required by mini-batch standard deviation tensor */
    algorithmFPType *stDev;                          /*!< Buffer that contains a block of values from mini-batch standard deviation tensor */
};

/**
 *  \brief Kernel for forward batch normalization layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class BatchNormalizationKernel : public Kernel
{
public:
    /* Computes the results of forward batch normalization layer */
    void compute(const batch_normalization::forward::Input *input,
                const batch_normalization::Parameter *parameter,
                batch_normalization::forward::Result *result);
protected:
    /* Computes mini-batch mean and variance of the input data over the specified dimension */
    void computeMeanAndVariance(
                const algorithmFPType *data, size_t offsetBefore, size_t dimensionSize, size_t offsetAfter,
                algorithmFPType *mean, algorithmFPType *stDev);

    /* Updates population mean and variance with the mini-batch mean and variance */
    void updatePopulationMeanAndVariance(const batch_normalization::forward::Input *input,
                size_t dimensionSize, algorithmFPType alpha,
                algorithmFPType *mean, algorithmFPType *stDev,
                batch_normalization::forward::Result *result);

    /* Computes batch normalization results */
    void computeResult(const batch_normalization::forward::Input *input,
                const algorithmFPType *data, size_t offsetBefore, size_t dimensionSize, size_t offsetAfter,
                const algorithmFPType *mean, const algorithmFPType *stDev,
                batch_normalization::forward::Result *result);
};

} // internal
} // forward
} // batch_normalization
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
