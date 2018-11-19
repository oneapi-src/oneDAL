/* file: xavier_initializer_impl.i */
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

/*
//++
//  Implementation of Xavier algorithm
//--
*/

#ifndef __XAVIER_INITIALIZER_IMPL_I__
#define __XAVIER_INITIALIZER_IMPL_I__

#include "initializers_impl.i"

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

using namespace daal::algorithms::distributions::uniform::internal;

template<typename algorithmFPType, Method method, CpuType cpu>
Status XavierKernel<algorithmFPType, method, cpu>::compute(const XavierInitializerTaskDescriptor &desc)
{
    initializers::internal::EngineImpl<cpu> engine(desc.engine);
    DAAL_CHECK_MALLOC(engine.get());

    size_t fanIn;
    size_t fanOut;
    Status status;
    DAAL_CHECK_STATUS(status, getFanInAndFanOut<algorithmFPType>(desc, fanIn, fanOut));

    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultSubtensor(desc.result, 0, 0, 0, desc.result->getDimensions()[0]);
    DAAL_CHECK_BLOCK_STATUS(resultSubtensor);
    algorithmFPType *resultArray = resultSubtensor.get();

    algorithmFPType scale = daal::internal::Math<double, cpu>::sSqrt(6.0 / ((double)fanIn + (double)fanOut));
    return UniformKernelDefault<algorithmFPType, cpu>::compute(-scale, scale, *engine, desc.result->getSize(), resultArray);
}

} // internal
} // namespace xavier
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
