/* file: batch_normalization_layer_forward_task_descriptor.cpp */
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
