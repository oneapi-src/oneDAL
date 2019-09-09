/* file: batch_normalization_layer_forward_task_descriptor.h */
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

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_TASK_DESCRIPTOR_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_TASK_DESCRIPTOR_H__

#include "neural_networks/layers/batch_normalization/batch_normalization_layer_forward.h"
#include "neural_networks/layers/batch_normalization/batch_normalization_layer_forward_types.h"

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

class BatchNormalizationTaskDescriptor
{
public:
    BatchNormalizationTaskDescriptor(Input *in, Result *re, Parameter *pa);

    data_management::Tensor *input;
    data_management::Tensor *weights;
    data_management::Tensor *biases;
    data_management::Tensor *inPopMean;
    data_management::Tensor *inPopVariance;
    data_management::Tensor *value;
    data_management::Tensor *auxMean;
    data_management::Tensor *auxStd;
    data_management::Tensor *auxPopMean;
    data_management::Tensor *auxPopVariance;
    Parameter *parameter;
};

} // internal
} // forward
} // batch_normalization
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
