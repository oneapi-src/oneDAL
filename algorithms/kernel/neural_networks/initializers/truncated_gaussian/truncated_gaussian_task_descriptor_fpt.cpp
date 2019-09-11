/* file: truncated_gaussian_task_descriptor_fpt.cpp */
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

#include "truncated_gaussian_task_descriptor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace truncated_gaussian
{
namespace internal
{

template<typename algorithmFPType>
TruncatedGaussianInitializerTaskDescriptor<algorithmFPType>::
TruncatedGaussianInitializerTaskDescriptor(Result *re, Parameter<algorithmFPType> *pa)
{
    a      = pa->a;
    b      = pa->b;
    mean   = pa->mean;
    sigma  = pa->sigma;
    layer  = pa->layer.get();
    engine = pa->engine.get();
    result = re->get(initializers::value).get();
}

template class TruncatedGaussianInitializerTaskDescriptor<DAAL_FPTYPE>;

} // internal
} // truncated_gaussian
} // initializers
} // neural_networks
} // algorithms
} // daal
