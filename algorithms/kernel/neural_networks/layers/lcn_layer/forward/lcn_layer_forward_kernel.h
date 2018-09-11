/* file: lcn_layer_forward_kernel.h */
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
//  Declaration of template function that calculate local contrast normalization.
//--


#ifndef __LCN_LAYER_FORWARD_KERNEL_H__
#define __LCN_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/lcn/lcn_layer.h"
#include "neural_networks/layers/lcn/lcn_layer_types.h"
#include "kernel.h"
#include "service_math.h"
#include "service_tensor.h"
#include "service_numeric_table.h"
#include "convolution2d_layer_forward_kernel.h"

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
 *  \brief Kernel for lcn calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class LCNKernel : public Kernel
{
public:
    services::Status compute(const Tensor &inputTensor, Tensor &sigmaTensor, Tensor &cTensor, Tensor &resultTensor, Tensor &centeredDataTensor,
                                                    Tensor &invMaxTensor, const lcn::Parameter &parameter, const Tensor &kernelTensor);
    services::Status initialize(const Tensor &inputTensor, Tensor &cTensor, Tensor &invMaxTensor, const lcn::Parameter &parameter, const Tensor &kernelTensor);
    services::Status reset();
protected:
    void getConvolutionWeightsFromInputKernel(algorithmFPType *weights, const lcn::Parameter &parameter);
    services::Status calculateCenteredDataAndSigma(const lcn::Parameter &parameter, const Tensor &inputTensor);
    services::Status calculateCTensorAndGetMaxArray(Tensor &cTensor);
    void calculateResult();
private:
    size_t batchDimension;
    size_t initialFirstDim;
    size_t initialSecondDim;
    size_t initialSumDimension;
    size_t nInputRows;
    size_t nSigmaRows;
    size_t nCRows;
    size_t nKernelRows;
    const algorithmFPType *inputArray;
    algorithmFPType *resultArray;
    algorithmFPType *centeredDataArray;
    algorithmFPType *sigmaArray;
    algorithmFPType *cArray;
    algorithmFPType *invMaxArray;
    const algorithmFPType *kernelArray;
    convolution2d::Parameter convParameter;

    services::Collection<size_t> dataDims;
    services::Collection<size_t> kernelDims;
    services::Collection<size_t> weightsDims;
    services::Collection<size_t> sDims;

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
    size_t dimsArray[4];
    size_t nKernels;

    convolution2d::forward::internal::Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> convKernel;
};
} // internal
} // forward

} // lcn
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
