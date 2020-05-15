/* file: spatial_stochastic_pooling2d_layer_forward_dense_default_batch_fpt_cpu.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
} // namespace interface1
} // namespace forward

} // namespace spatial_stochastic_pooling2d

namespace spatial_pooling2d
{
namespace forward
{
namespace internal
{
template class PoolingKernel<DAAL_FPTYPE, spatial_pooling2d::internal::stochastic, DAAL_CPU>;
} // namespace internal
} // namespace forward
} // namespace spatial_pooling2d

} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
