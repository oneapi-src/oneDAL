/* file: lcn_layer_backward_kernel.h */
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
//  Declaration of template function that calculate local contrast normalization.
//--


#ifndef __LCN_LAYER_BACKWARD_KERNEL_H__
#define __LCN_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/lcn/lcn_layer.h"
#include "neural_networks/layers/lcn/lcn_layer_types.h"
#include "convolution2d_layer_backward.h"
#include "kernel.h"
#include "service_math.h"
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
namespace lcn
{
namespace backward
{
namespace internal
{
/**
 * \brief Structure for storing data used in itermediate computations
 *        in backward local contrast normalization layer
 */
template<typename algorithmFPType, Method method, CpuType cpu>
struct LCNTask
{
    /*
     * Constructs the structure for storing data used in itermediate computations
     * in backward local contrast normalization layer
     */
    LCNTask(Tensor *auxCenteredDataTensor, Tensor *auxSigmaTensor,
            Tensor *auxCTensor, Tensor *auxInvMaxTensor, Tensor *kernelTensor, Tensor *inGradTensor,
            Tensor *gradientTensor, const lcn::Parameter *parameter);

    virtual ~LCNTask() {};

    services::Collection<size_t> dataDims;
    services::Collection<size_t> kernelDims;

    size_t nSigmaElements;
    size_t nDataElements;
    size_t nKernelElements;
    size_t nWeightsElements;
    size_t nCElements;

    size_t dataOffsetBeforeDim;
    size_t dataOffsetAfterDim;

    size_t sumDimension;
    size_t firstDim;
    size_t secondDim;
    size_t nDims;

    size_t weightsSecondDim;
    double sigmaThreshold;

    const algorithmFPType *inGradArray;
    const algorithmFPType *auxCDArray;
    const algorithmFPType *auxSigmaArray;
    const algorithmFPType *auxInvMaxArray;
    const algorithmFPType *auxCArray;
    const algorithmFPType *kernelArray;
    algorithmFPType *gradientArray;

    convolution2d::Parameter convParameter;

    ReadSubtensor<algorithmFPType, cpu, Tensor> inGradBlock;
    ReadSubtensor<algorithmFPType, cpu, Tensor> cdBlock;
    ReadSubtensor<algorithmFPType, cpu, Tensor> cBlock;
    ReadSubtensor<algorithmFPType, cpu, Tensor> sigmaBlock;
    ReadSubtensor<algorithmFPType, cpu, Tensor> invMaxBlock;
    ReadSubtensor<algorithmFPType, cpu, Tensor> kernelBlock;
    WriteSubtensor<algorithmFPType, cpu, Tensor> gradientBlock;
};
/**
 *  \brief Kernel for lcn calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class LCNKernel : public Kernel
{
public:
    void compute(LCNTask<algorithmFPType, method, cpu> &task, const lcn::Parameter *parameter);
protected:
    void prepareGDivGSqAndGConv(LCNTask<algorithmFPType, method, cpu> &task, algorithmFPType *tempArray, algorithmFPType *gSqTempOfSigmaSizeArray);
    void computeTwoConvAndFinalResult(LCNTask<algorithmFPType, method, cpu> &task, const lcn::Parameter *parameter, algorithmFPType *gConvTempArrayOfSigmaSize,
                                      algorithmFPType *gSqTempOfSigmaSizeArray);
    void getConvolutionWeightsFromInputKernel(algorithmFPType *weightsArray, LCNTask<algorithmFPType, method, cpu> &task, const lcn::Parameter *parameter);
};

} // internal
} // backward
} // lcn
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
