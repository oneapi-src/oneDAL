/* file: stochastic_pooling2d_layer_backward_dense_default_batch_fpt_cpu.cpp */
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


#include "stochastic_pooling2d_layer_backward_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace stochastic_pooling2d
{

namespace backward
{
namespace interface1
{
template class neural_networks::layers::stochastic_pooling2d::backward::BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // interface1
} // backward

}
}
}
}
}
