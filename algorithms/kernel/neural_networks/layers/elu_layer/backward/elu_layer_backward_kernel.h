/* file: elu_layer_backward_kernel.h */
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
//  Declaration of template function that calculate relus.
//--


#ifndef __ELU_LAYER_BACKWARD_KERNEL_H__
#define __ELU_LAYER_BACKWARD_KERNEL_H__

#include "kernel.h"
#include "elu_common.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace backward
{
namespace internal
{

using namespace daal::services;
using namespace daal::data_management;

using elu::internal::BlockSizeType;
using elu::internal::ScalableTlsBuffer;

/**
 *  \brief Kernel for ELU calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class ELUKernel : public Kernel
{
private:
    ScalableTlsBuffer<algorithmFPType, cpu> _intermediateValuesTls;
    ScalableTlsBuffer<BlockSizeType, cpu> _indicesTls;

public:
    ELUKernel() : _intermediateValuesTls( elu::internal::getMaxBlockSize<algorithmFPType, cpu>() ),
                  _indicesTls( elu::internal::getMaxBlockSize<algorithmFPType, cpu>() ) { }

    Status compute(const Parameter &parameter,
                   const Tensor &inputGradientTensor,
                   const Tensor &auxDataTensor,
                   const Tensor *auxValueTensor,
                         Tensor &gradientTensor);

private:

    Status computeLayoutAgnostic(const Tensor &inputGradientTensor,
                                 const Tensor &auxDataTensor,
                                 const Tensor &auxValueTensor,
                                       Tensor &gradientTensor);

    Status computeInMklLayout(const Tensor &inputGradientTensor,
                              const Tensor &auxDataTensor,
                              const Tensor &auxValueTensor,
                                    Tensor &gradientTensor);

    void computeInRawLayout(const algorithmFPType *inputGradient,
                            const algorithmFPType *auxData,
                            const algorithmFPType *auxValue,
                                  algorithmFPType *gradient,
                                  size_t dataSize);

    void computeBlock(const algorithmFPType *inputGradient,
                      const algorithmFPType *auxData,
                      const algorithmFPType *auxValue,
                            algorithmFPType *gradient,
                            size_t blockSize);

    Status computeWithoutAuxValues(const Tensor &inputGradientTensor,
                                   const Tensor &auxDataTensor,
                                         Tensor &gradientTensor,
                                         algorithmFPType alpha);

    void computeBlockWithoutAuxValues(const algorithmFPType *inputGradient,
                                      const algorithmFPType *auxData,
                                            algorithmFPType *gradient,
                                            algorithmFPType alpha,
                                            size_t blockSize);

};

} // internal
} // backward
} // elu
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
