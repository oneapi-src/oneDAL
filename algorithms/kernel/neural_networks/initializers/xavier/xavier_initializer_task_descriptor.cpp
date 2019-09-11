/* file: xavier_initializer_task_descriptor.cpp */
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

#include "xavier_initializer_task_descriptor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace xavier
{
namespace internal
{

XavierInitializerTaskDescriptor::XavierInitializerTaskDescriptor(Result *re, Parameter *pa)
{
    engine = pa->engine.get();
    layer  = pa->layer.get();
    result = re->get(initializers::value).get();
}

} // internal
} // xavier
} // initializers
} // neural_networks
} // algorithms
} // daal
