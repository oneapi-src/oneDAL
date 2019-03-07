/* file: spatial_stochastic_pooling2d_layer_forward_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of forward pooling layer.
//--


#include "spatial_stochastic_pooling2d_layer_forward_batch_container.h"
#include "spatial_pooling2d_layer_forward_kernel.h"
#include "spatial_pooling2d_layer_forward_impl.i"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_stochastic_pooling2d
{

namespace forward
{
namespace interface1
{
template class neural_networks::layers::spatial_stochastic_pooling2d::forward::BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // interface1
} // forward

}

namespace spatial_pooling2d
{
namespace forward
{
namespace internal
{
template class PoolingKernel<DAAL_FPTYPE, spatial_pooling2d::internal::stochastic, DAAL_CPU>;
} // internal
}
}

}
}
}
}
