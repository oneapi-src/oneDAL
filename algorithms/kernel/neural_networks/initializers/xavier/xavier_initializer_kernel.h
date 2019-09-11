/* file: xavier_initializer_kernel.h */
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
//  Declaration of template function that calculate Xavier.
//--

#ifndef __XAVIER_INITIALIZER_KERNEL_H__
#define __XAVIER_INITIALIZER_KERNEL_H__

#include "kernel.h"
#include "service_math.h"
#include "layer_forward.h"
#include "layer_forward_types.h"
#include "service_math.h"
#include "service_tensor.h"
#include "uniform_kernel.h"

#include "neural_networks/initializers/xavier/xavier_initializer.h"
#include "neural_networks/initializers/xavier/xavier_initializer_types.h"

#include "xavier_initializer_misc.h"
#include "xavier_initializer_task_descriptor.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

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

/**
 *  \brief Kernel for xavier calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class XavierKernel : public Kernel
{
public:
    Status compute(const XavierInitializerTaskDescriptor &desc);
};

} // internal
} // xavier
} // initializers
} // neural_networks
} // algorithms
} // daal

#endif
