/* file: truncated_gaussian_kernel.h */
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
//  Declaration of template function that implements truncated gaussian initializer.
//--

#ifndef __TRUNCATED_GAUSSIAN_INITIALIZER_KERNEL_H__
#define __TRUNCATED_GAUSSIAN_INITIALIZER_KERNEL_H__

#include "kernel.h"
#include "service_math.h"
#include "service_tensor.h"
#include "threading.h"
#include "uniform_kernel.h"

#include "neural_networks/initializers/truncated_gaussian/truncated_gaussian_initializer.h"
#include "neural_networks/initializers/truncated_gaussian/truncated_gaussian_initializer_types.h"

#include "truncated_gaussian_task_descriptor.h"

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
namespace truncated_gaussian
{
namespace internal
{

/**
 *  \brief Kernel for truncated_gaussian calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TruncatedGaussianKernel : public Kernel
{
public:
    Status compute(const TruncatedGaussianInitializerTaskDescriptor<algorithmFPType> &desc);

private:
    algorithmFPType getCDFNormal(algorithmFPType p, algorithmFPType mean, algorithmFPType sigma);
    const size_t _nElemsInBlock = 1000;
};

} // internal
} // truncated_gaussian
} // initializers
} // neural_networks
} // algorithms
} // daal

#endif
