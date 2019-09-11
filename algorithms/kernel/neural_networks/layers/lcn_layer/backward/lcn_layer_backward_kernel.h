/* file: lcn_layer_backward_kernel.h */
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
#include "convolution2d_layer_backward_kernel.h"
#include "threading.h"
#include "service_memory.h"
#include "layers_threading.h"

using namespace daal::algorithms::neural_networks::layers::convolution2d::backward::internal;
using namespace daal::data_management;
using namespace daal::services;
using namespace daal::services::internal;
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
 *  \brief Kernel for lcn calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class LCNKernel : public Kernel
{
public:
    services::Status compute(const Tensor &auxCenteredDataTensor, const Tensor &auxSigmaTensor, const Tensor &auxCTensor,
                                                      const Tensor &auxInvMaxTensor, const Tensor &kernelTensor, const Tensor &inGradTensor,
                                                      Tensor &gradientTensor, const lcn::Parameter &parameter);
    services::Status initialize(const Tensor &auxCenteredDataTensor, const Tensor &auxSigmaTensor, const Tensor &auxCTensor,
                                                         const Tensor &kernelTensor, const lcn::Parameter &parameter);
    services::Status reset();

private:
    size_t nDims;
    size_t nDataRows;
    size_t nSigmaRows;
    size_t nCRows;
    size_t nKernelRows;

    services::Collection<size_t> dataDims;
    services::Collection<size_t> kernelDims;
    Collection<size_t> sigmaDims;

    size_t nDataElements;
    size_t nKernelElements;
    size_t nWeightsElements;
    size_t nCElements;

    size_t dataOffsetBeforeDim;
    size_t dataOffsetAfterDim;

    size_t batchDimension;
    size_t sumDimension;
    size_t firstDim;
    size_t secondDim;

    size_t initialFirstDim;
    size_t initialSecondDim;
    size_t initialSumDimension;
    size_t fDimN;

    double sigmaThreshold;

    convolution2d::Parameter convParameter;

    void getFixedDimsIndexes(size_t *fDims, size_t i);
};

} // internal
} // backward
} // lcn
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
