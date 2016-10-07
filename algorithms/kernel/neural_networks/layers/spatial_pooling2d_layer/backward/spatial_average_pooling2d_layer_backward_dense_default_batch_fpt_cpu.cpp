/* file: spatial_average_pooling2d_layer_backward_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of backward pooling layer.
//--


#include "spatial_average_pooling2d_layer_backward_batch_container.h"
#include "spatial_pooling2d_layer_backward_kernel.h"
#include "spatial_pooling2d_layer_backward_impl.i"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_average_pooling2d
{

namespace backward
{
namespace interface1
{
template class neural_networks::layers::spatial_average_pooling2d::backward::BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // interface1
} // backward

}

namespace spatial_pooling2d
{
namespace backward
{
namespace internal
{
template class PoolingKernel<DAAL_FPTYPE, spatial_pooling2d::internal::average, DAAL_CPU>;
} // internal
}
}

}
}
}
}
