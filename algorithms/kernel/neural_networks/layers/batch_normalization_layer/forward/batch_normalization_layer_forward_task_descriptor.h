/* file: batch_normalization_layer_forward_task_descriptor.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
