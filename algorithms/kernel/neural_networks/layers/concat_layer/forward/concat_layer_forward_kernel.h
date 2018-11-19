/* file: concat_layer_forward_kernel.h */
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


#ifndef __CONCAT_LAYER_FORWARD_KERNEL_H__
#define __CONCAT_LAYER_FORWARD_KERNEL_H__

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
namespace forward
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
    services::Status compute(size_t nInputs, Tensor *inputTensors[], const concat::Parameter *parameter,
                 Tensor *resultTensor);

    ~ConcatKernel()
    {
        if (concatPrim)
        {
            dnn::xDelete(concatPrim);
        }
        if (inputLayouts)
        {
            delete [] inputLayouts;
        }
    }
private:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;

    const size_t _nRowsInBlock = 5000;

    dnnPrimitive_t concatPrim = NULL;
    dnnLayout_t *inputLayouts = NULL;
};
} // internal
} // forward
} // concat
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
