/* file: neural_networks_training_dense_feedforward_batch_fpt_cpu.cpp */
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
//  Implementation of neural networks calculation functions for AVX2.
//--


#include "neural_networks_training_batch_container.h"
#include "neural_networks_training_feedforward_kernel.h"
#include "neural_networks_training_feedforward_impl.i"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{
namespace interface1
{
template class neural_networks::training::BatchContainer<DAAL_FPTYPE, feedforwardDense, DAAL_CPU>;
}
namespace internal
{
template class NeuralNetworksFeedforwardTrainingKernel<DAAL_FPTYPE, feedforwardDense, DAAL_CPU>;
}
}
}
}
}
