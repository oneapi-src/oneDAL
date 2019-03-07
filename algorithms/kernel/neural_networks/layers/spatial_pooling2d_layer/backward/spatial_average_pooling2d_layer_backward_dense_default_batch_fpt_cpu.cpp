/* file: spatial_average_pooling2d_layer_backward_dense_default_batch_fpt_cpu.cpp */
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
