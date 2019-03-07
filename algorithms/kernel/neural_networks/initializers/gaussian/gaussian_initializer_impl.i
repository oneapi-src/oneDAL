/* file: gaussian_initializer_impl.i */
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

/*
//++
//  Implementation of gaussian algorithm
//--
*/

#ifndef __GAUSSIAN_INITIALIZER_IMPL_I__
#define __GAUSSIAN_INITIALIZER_IMPL_I__

#include "initializers_impl.i"

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

template<typename algorithmFPType, Method method, CpuType cpu>
Status GaussianKernel<algorithmFPType, method, cpu>::compute(const GaussianInitializerTaskDescriptor &desc)
{
    initializers::internal::EngineImpl<cpu> engine(desc.engine);
    DAAL_CHECK_MALLOC(engine.get());

    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultSubtensor(desc.result, 0, 0, 0, desc.result->getDimensionSize(0));
    DAAL_CHECK_BLOCK_STATUS(resultSubtensor);
    algorithmFPType *resultArray = resultSubtensor.get();

    size_t size = desc.result->getSize();

    distributions::normal::Parameter<algorithmFPType> normalParameter((algorithmFPType)desc.a, (algorithmFPType)desc.sigma);
    return distributions::normal::internal::NormalKernelDefault<algorithmFPType, cpu>::compute(
        &normalParameter, *engine, size, resultArray);
}

} // internal
} // namespace gaussian
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
