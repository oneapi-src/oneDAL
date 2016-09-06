/* file: batch_normalization_layer_backward_kernel.h */
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
//  Declaration of template function that calculate backward batch normalization layer relults.
//--


#ifndef __BATCH_NORMALIZATION_LAYER_BACKWARD_KERNEL_H__
#define __BATCH_NORMALIZATION_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/batch_normalization/batch_normalization_layer_backward.h"
#include "neural_networks/layers/batch_normalization/batch_normalization_layer_backward_types.h"
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
namespace backward
{
namespace internal
{
/**
 * \brief Structure for storing data used in itermediate computations
 *        in backward batch normalization layer
 */
template<typename algorithmFPType, Method method, CpuType cpu>
struct BatchNormalizationTask
{
    /*
     * Constructs the structure for storing data used in itermediate computations
     * in backward batch normalization layer
     */
    BatchNormalizationTask(const batch_normalization::backward::Input *input,
            batch_normalization::backward::Result *result, size_t dimension,
            services::SharedPtr<services::KernelErrorCollection> _errors);

    virtual ~BatchNormalizationTask();

    /**
     * p-dimensional tensor of size n_1 * n_2 * ... * n_p that stores
     * backward batch normalization layer input data that is a gradient computed on the prior layer
     */
    SharedPtr<Tensor> inputGradientTensor;
    SubtensorDescriptor<algorithmFPType> inputGradientBlock;    /*!< Object that manages read operations required by input gradient tensor */
    algorithmFPType *inputGradient;                             /*!< Buffer that contains a block of values from input gradient tensor */


    /**
     * p-dimensional tensor of size n_1 * n_2 * ... * n_p that stores
     * forward batch normalization layer input data
     */
    SharedPtr<Tensor> dataTensor;
    SubtensorDescriptor<algorithmFPType> dataBlock;    /*!< Object that manages read operations required by input data tensor */
    algorithmFPType *data;                             /*!< Buffer that contains a block of values from input data tensor */

    /*
     * Input gradient tensor is viewed by the algorithm as a 3-dimensional tensor of size offsetBefore * dimensionSize * offsetAfter
     */
    size_t offsetBefore;    /*!< n_1 * ... * n_(k-1), where k is the dimension over which the normalization is performed */
    size_t dimensionSize;   /*!< n_k - size of the dimension over which the normalization is performed */
    size_t offsetAfter;     /*!< n_(k+1) * ... * n_p */

    SharedPtr<Tensor> meanTensor;                    /*!< 1-dimensional tensor of size n_k that stores mini-batch mean */
    SubtensorDescriptor<algorithmFPType> meanBlock;  /*!< Object that manages read operations required by mini-batch mean tensor */
    algorithmFPType *mean;                           /*!< Buffer that contains a block of values from mini-batch mean tensor */
    algorithmFPType *invStDev;                       /*!< Buffer that contains a block of inverse mini-batch standard deviation values */

    SharedPtr<Tensor> weightsDerTensor;                     /*!< 1-dimensional tensor of size n_k that stores weights derivative */
    SubtensorDescriptor<algorithmFPType> weightsDerBlock;   /*!< Object that manages write operations required by weights derivative tensor */
    algorithmFPType *weightsDer;                            /*!< Buffer that contains a block of values from weights derivative tensor */

    SharedPtr<Tensor> biasesDerTensor;                      /*!< 1-dimensional tensor of size n_k that stores biases derivative */
    SubtensorDescriptor<algorithmFPType> biasesDerBlock;    /*!< Object that manages write operations required by biases derivative tensor */
    algorithmFPType *biasesDer;                             /*!< Buffer that contains a block of values from biases derivative tensor */
};

/**
 *  \brief Kernel for backward batch normalization layer results computation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class BatchNormalizationKernel : public Kernel
{
public:
    void compute(const batch_normalization::backward::Input *input,
                const batch_normalization::Parameter *parameter,
                batch_normalization::backward::Result *result);

protected:
    void computeWeightsAndBiasesDerivatives(const algorithmFPType *inputGradient, const algorithmFPType *data,
                size_t offsetBefore, size_t dimensionSize, size_t offsetAfter,
                const algorithmFPType *mean, const algorithmFPType *invStDev,
                algorithmFPType *weightsDer, algorithmFPType *biasesDer);

    void computeGradient(const batch_normalization::backward::Input *input,
                const algorithmFPType *inputGradient, const algorithmFPType *data,
                size_t offsetBefore, size_t dimensionSize, size_t offsetAfter,
                const algorithmFPType *mean, const algorithmFPType *invStDev,
                const algorithmFPType *weightsDer, const algorithmFPType *biasesDer,
                batch_normalization::backward::Result *result);
};

} // internal
} // backward
} // batch_normalization
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
