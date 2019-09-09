/* file: batch_normalization_layer_forward_task_descriptor.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "batch_normalization_layer_forward_task_descriptor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace forward
{
namespace internal
{

BatchNormalizationTaskDescriptor::BatchNormalizationTaskDescriptor(
    Input *in, Result *re, Parameter *pa)
{
    parameter      = pa;
    input          = in->get ( layers::forward::data                      ).get();
    weights        = in->get ( layers::forward::weights                   ).get();
    biases         = in->get ( layers::forward::biases                    ).get();
    inPopMean      = in->get ( forward::populationMean                    ).get();
    inPopVariance  = in->get ( forward::populationVariance                ).get();
    value          = re->get ( layers::forward::value                     ).get();
    auxMean        = re->get ( batch_normalization::auxMean               ).get();
    auxStd         = re->get ( batch_normalization::auxStandardDeviation  ).get();
    auxPopMean     = re->get ( batch_normalization::auxPopulationMean     ).get();
    auxPopVariance = re->get ( batch_normalization::auxPopulationVariance ).get();
}

} // internal
} // forward
} // batch_normalization
} // layers
} // neural_networks
} // algorithms
} // daal
