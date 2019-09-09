/* file: mt2203.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

//++
//  Implementation of mt2203 engine
//--

#include "algorithms/engines/mt2203/mt2203.h"
#include "mt2203_batch_impl.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mt2203
{
namespace interface1
{

using namespace daal::services;
using namespace mt2203::internal;

template<typename algorithmFPType, Method method>
SharedPtr<Batch<algorithmFPType, method> > Batch<algorithmFPType, method>::create(size_t seed, services::Status *st)
{
    SharedPtr<Batch<algorithmFPType, method> > engPtr;

    int cpuid = (int)Environment::getInstance()->getCpuId();
        switch(cpuid)
        {
#ifdef DAAL_KERNEL_AVX512
            case avx512: DAAL_KERNEL_AVX512_ONLY_CODE(engPtr.reset(new BatchImpl<avx512, algorithmFPType, method>(seed, st))); break;
#endif
#ifdef DAAL_KERNEL_AVX512_MIC
            case avx512_mic: DAAL_KERNEL_AVX512_MIC_ONLY_CODE(engPtr.reset(new BatchImpl<avx512_mic, algorithmFPType, method>(seed, st))); break;
#endif
#ifdef DAAL_KERNEL_AVX2
            case avx2: DAAL_KERNEL_AVX2_ONLY_CODE(engPtr.reset(new BatchImpl<avx2, algorithmFPType, method>(seed, st))); break;
#endif
#ifdef DAAL_KERNEL_AVX
            case avx: DAAL_KERNEL_AVX_ONLY_CODE(engPtr.reset(new BatchImpl<avx, algorithmFPType, method>(seed, st))); break;
#endif
#ifdef DAAL_KERNEL_SSE42
            case sse42: DAAL_KERNEL_SSE42_ONLY_CODE(engPtr.reset(new BatchImpl<sse42, algorithmFPType, method>(seed, st))); break;
#endif
#ifdef DAAL_KERNEL_SSSE3
            case ssse3: DAAL_KERNEL_SSSE3_ONLY_CODE(engPtr.reset(new BatchImpl<ssse3, algorithmFPType, method>(seed, st))); break;
#endif
            default: engPtr.reset(new BatchImpl<sse2, algorithmFPType, method>(seed, st)); break;
        };
    return engPtr;
}

template class Batch<double, defaultDense>;
template class Batch<float, defaultDense>;

} // namespace interface1
} // namespace mt2203
} // namespace engines
} // namespace algorithms
} // namespace daal
