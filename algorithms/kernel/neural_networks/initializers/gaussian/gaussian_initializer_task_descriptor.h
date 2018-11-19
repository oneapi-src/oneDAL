/* file: gaussian_initializer_task_descriptor.h */
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

#ifndef __GAUSSIAN_INITIALIZER_TASK_DESCRIPTOR_H__
#define __GAUSSIAN_INITIALIZER_TASK_DESCRIPTOR_H__

#include "neural_networks/initializers/gaussian/gaussian_initializer.h"
#include "neural_networks/initializers/gaussian/gaussian_initializer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace gaussian
{
namespace internal
{

class GaussianInitializerTaskDescriptor
{
public:
    GaussianInitializerTaskDescriptor(Result *re, Parameter *pa);

    engines::BatchBase          *engine;
    data_management::Tensor     *result;
    layers::forward::LayerIface *layer;
    double a;
    double sigma;
};

} // internal
} // gaussian
} // initializers
} // neural_networks
} // algorithms
} // daal

#endif
