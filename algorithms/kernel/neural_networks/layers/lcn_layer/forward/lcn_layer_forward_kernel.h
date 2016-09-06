/* file: lcn_layer_forward_kernel.h */
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


#ifndef __LCN_LAYER_FORWARD_KERNEL_H__
#define __LCN_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/lcn/lcn_layer.h"
#include "neural_networks/layers/lcn/lcn_layer_types.h"
#include "convolution2d_layer_forward.h"
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
namespace forward
{
namespace internal
{
/**
 * \brief Structure for storing data used in itermediate computations
 *        in forward local contrast normalization layer
 */
template<typename algorithmFPType, Method method, CpuType cpu>
struct LCNTask
{
    /*
     * Constructs the structure for storing data used in itermediate computations
     * in forward local contrast normalization layer
     */
    LCNTask(Tensor *inputTensor, Tensor *resultTensor, Tensor *centeredDataTensor,
            Tensor *sigmaTensor, Tensor *cTensor, Tensor *invMaxTensor, const lcn::Parameter *parameter,
            Tensor *kernelTensor);

    virtual ~LCNTask() {};

    services::Collection<size_t> dataDims;
    services::Collection<size_t> kernelDims;

    size_t nSigmaElements;
    size_t nDataElements;
    size_t nKernelElements;
    size_t nWeightsElements;
    size_t dataOffsetBeforeDim;
    size_t dataOffsetAfterDim;

    size_t sumDimension;
    size_t firstDim;
    size_t secondDim;
    size_t weightsSecondDim;

    convolution2d::Parameter convParameter;

    const algorithmFPType *inputArray;
    algorithmFPType *resultArray;
    algorithmFPType *centeredDataArray;
    algorithmFPType *sigmaArray;
    algorithmFPType *cArray;
    algorithmFPType *invMaxArray;
    const algorithmFPType *kernelArray;

    ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock;
    WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock;
    WriteSubtensor<algorithmFPType, cpu, Tensor> cdBlock;
    WriteSubtensor<algorithmFPType, cpu, Tensor> cBlock;
    WriteSubtensor<algorithmFPType, cpu, Tensor> sigmaBlock;
    WriteSubtensor<algorithmFPType, cpu, Tensor> invMaxBlock;
    ReadSubtensor<algorithmFPType, cpu, Tensor> kernelBlock;
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
    void getConvolutionWeightsFromInputKernel(algorithmFPType *weights, LCNTask<algorithmFPType, method, cpu> &task, const lcn::Parameter *parameter);
    void calculateCenteredDataAndSigma(LCNTask<algorithmFPType, method, cpu> &task, const lcn::Parameter *parameter);
    void calculateCTensorAndGetMaxArray(LCNTask<algorithmFPType, method, cpu> &task);
    void calculateResult(LCNTask<algorithmFPType, method, cpu> &task);
};
} // internal
} // forward

} // lcn
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
