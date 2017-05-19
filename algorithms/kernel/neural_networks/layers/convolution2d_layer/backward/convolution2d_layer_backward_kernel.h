/* file: convolution2d_layer_backward_kernel.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
    const convolution2d::Parameter *parameter, Tensor *wDerTensor, Tensor *bDerTensor, Tensor *resultTensor);

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
