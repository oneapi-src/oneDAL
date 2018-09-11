/* file: mcg59.cpp */
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

//++
//  Implementation of mcg59 engine
//--

#include "algorithms/engines/mcg59/mcg59.h"
#include "mcg59_batch_impl.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mcg59
{
namespace interface1
{

using namespace daal::services;
using namespace mcg59::internal;

template<typename algorithmFPType, Method method>
SharedPtr<Batch<algorithmFPType, method> > Batch<algorithmFPType, method>::create(size_t seed)
{
    SharedPtr<Batch<algorithmFPType, method> > engPtr;

    int cpuid = (int)Environment::getInstance()->getCpuId();
        switch(cpuid)
        {
#ifdef DAAL_KERNEL_AVX512
            case avx512: DAAL_KERNEL_AVX512_ONLY_CODE(engPtr.reset(new BatchImpl<avx512, algorithmFPType, method>(seed))); break;
#endif
#ifdef DAAL_KERNEL_AVX512_mic
            case avx512_mic: DAAL_KERNEL_AVX512_mic_ONLY_CODE(engPtr.reset(new BatchImpl<avx512_mic, algorithmFPType, method>(seed))); break;
#endif
#ifdef DAAL_KERNEL_AVX2
            case avx2: DAAL_KERNEL_AVX2_ONLY_CODE(engPtr.reset(new BatchImpl<avx2, algorithmFPType, method>(seed))); break;
#endif
#ifdef DAAL_KERNEL_AVX
            case avx: DAAL_KERNEL_AVX_ONLY_CODE(engPtr.reset(new BatchImpl<avx, algorithmFPType, method>(seed))); break;
#endif
#ifdef DAAL_KERNEL_SSE42
            case sse42: DAAL_KERNEL_SSE42_ONLY_CODE(engPtr.reset(new BatchImpl<sse42, algorithmFPType, method>(seed))); break;
#endif
#ifdef DAAL_KERNEL_SSSE3
            case ssse3: DAAL_KERNEL_SSSE3_ONLY_CODE(engPtr.reset(new BatchImpl<ssse3, algorithmFPType, method>(seed))); break;
#endif
            default: engPtr.reset(new BatchImpl<sse2, algorithmFPType, method>(seed)); break;
        };
    return engPtr;
}

template class Batch<double, defaultDense>;
template class Batch<float, defaultDense>;

} // namespace interface1
} // namespace mcg59
} // namespace engines
} // namespace algorithms
} // namespace daal
