/* file: concat_layer_backward_kernel.h */
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
//  Declaration of template function that calculate concats.
//--


#ifndef __CONCAT_LAYER_BACKWARD_KERNEL_H__
#define __CONCAT_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/concat/concat_layer.h"
#include "neural_networks/layers/concat/concat_layer_types.h"
#include "kernel.h"
#include "service_dnn.h"
#include "service_dnn_internal.h"
#include "layers_threading.h"

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
namespace concat
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for concat calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class ConcatKernel : public Kernel
{
public:
    services::Status compute(Tensor *inputTensor, const NumericTable *forwardOutputTable, const concat::Parameter *parameter, Tensor *resultTensors[]);

    ~ConcatKernel()
    {
        if (splitPrim)
        {
            dnn::xDelete(splitPrim);
        }
    }

private:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;

    const size_t _nRowsInBlock = 5000;

    dnnPrimitive_t splitPrim = NULL;
};

} // internal
} // backward
} // relu
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
