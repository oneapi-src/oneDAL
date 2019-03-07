/* file: elu_layer_forward_kernel.h */
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
//  Declaration of template function that calculate ELU.
//--


#ifndef __ELU_LAYER_FORWARD_KERNEL_H__
#define __ELU_LAYER_FORWARD_KERNEL_H__

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
namespace forward
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
                   const Tensor &dataTensor,
                         Tensor &valueTensor,
                         Tensor *auxValueTensor);

private:

    Status computeLayoutAgnostic(const Tensor &dataTensor,
                                       Tensor &valueTensor,
                                       Tensor *auxValueTensor,
                                       algorithmFPType alpha);

    Status computeInMKLLayout(const Tensor &dataTensor,
                                    Tensor &valueTensor,
                                    Tensor *auxValueTensor,
                                    algorithmFPType alpha);

    void computeInRawLayout(const algorithmFPType *data,
                                  algorithmFPType *value,
                                  algorithmFPType *auxValue,
                                  algorithmFPType alpha,
                                  size_t dataSize);

    void computeBlock(const algorithmFPType *data,
                            algorithmFPType *value,
                            algorithmFPType *auxValue,
                            algorithmFPType alpha,
                            size_t blockSize);

    void computeInRawLayoutPrediction(const algorithmFPType *data,
                                            algorithmFPType *value,
                                            algorithmFPType alpha,
                                            size_t dataSize);

    void computeBlockPrediction(const algorithmFPType *data,
                                      algorithmFPType *value,
                                      algorithmFPType alpha,
                                      size_t blockSize);
};

} // internal
} // forward
} // elu
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
