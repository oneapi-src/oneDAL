/* file: mt19937_batch_impl.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

/*
//++
//  Implementation of the class defining the mt19937 engine
//--
*/

#include "engines/mt19937/mt19937.h"
#include "engine_batch_impl.h"
#include "service_rng.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mt19937
{
namespace internal
{

template<CpuType cpu, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class BatchImpl : public algorithms::engines::mt19937::interface1::Batch<algorithmFPType, method>, public algorithms::engines::internal::BatchBaseImpl
{
public:
    BatchImpl(size_t seed = 777) : baseRng(seed, __DAAL_BRNG_MT19937) {}

    void *getState() DAAL_C11_OVERRIDE
    {
        return baseRng.getState();
    }

    services::Status saveStateImpl(byte* dest) DAAL_C11_OVERRIDE
    {
        baseRng.saveState((void *)dest);
        return services::Status();
    }

    services::Status loadStateImpl(const byte* src) DAAL_C11_OVERRIDE
    {
        baseRng.loadState((const void *)src);
        return services::Status();
    }

    ~BatchImpl() {}

protected:
    daal::internal::BaseRNGs<cpu> baseRng;
};

} // namespace interface1
} // namespace mt19937
} // namespace engines
} // namespace algorithms
} // namespace daal
