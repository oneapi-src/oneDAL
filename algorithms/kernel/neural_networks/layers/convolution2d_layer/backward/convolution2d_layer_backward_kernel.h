/* file: convolution2d_layer_backward_kernel.h */
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
//  Declaration of template function that calculate convolution2ds.
//--


#ifndef __CONVOLUTION2D_LAYER_BACKWARD_KERNEL_H__
#define __CONVOLUTION2D_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/convolution2d/convolution2d_layer.h"
#include "neural_networks/layers/convolution2d/convolution2d_layer_types.h"
#include "kernel.h"
#include "service_math.h"
#include "numeric_table.h"
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
namespace convolution2d
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for convolution2d calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class Convolution2dKernel : public Kernel
{
public:
    services::Status initialize(bool resultFlag = true, bool wDerFlag = true, bool bDerFlag = true);

    services::Status compute(Tensor *inGradTensor, Tensor *xTensor, Tensor *wTensor,
    const convolution2d::Parameter &parameter, Tensor *wDerTensor, Tensor *bDerTensor, Tensor *resultTensor);

    services::Status reset();

    ~Convolution2dKernel()
    {
        if (convGrad)
        {
            dnn::xDelete(convGrad);
        }
        if (convBias)
        {
            dnn::xDelete(convBias);
        }
        if (convFilt)
        {
            dnn::xDelete(convFilt);
        }
    }

private:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;

    bool _resultFlag;
    bool _wDerFlag;
    bool _bDerFlag;

    dnnPrimitive_t convGrad = NULL;
    dnnPrimitive_t convFilt = NULL;
    dnnPrimitive_t convBias = NULL;
};

} // internal
} // backward
} // convolution2d
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
