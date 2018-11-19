/* file: mt19937_batch_impl.h */
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
//  Implementation of the class defining the mt19937 engine
//--
*/

#include "engines/mt19937/mt19937.h"
#include "engine_batch_impl.h"
#include "service_rng.h"
#include "service_numeric_table.h"

static const int leapfrogMethodErrcode  = -1002;
static const int skipAheadMethodErrcode = -1003;

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
    typedef algorithms::engines::mt19937::interface1::Batch<algorithmFPType, method> super1;
    typedef algorithms::engines::internal::BatchBaseImpl super2;
    BatchImpl(size_t seed = 777) : baseRng(seed, __DAAL_BRNG_MT19937), super2(seed) {}

    void *getState() DAAL_C11_OVERRIDE
    {
        return baseRng.getState();
    }

    int getStateSize() const DAAL_C11_OVERRIDE
    {
        return baseRng.getStateSize();
    }

    services::Status saveStateImpl(byte* dest) const DAAL_C11_OVERRIDE
    {
        DAAL_CHECK(!baseRng.saveState((void *)dest), ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

    services::Status loadStateImpl(const byte* src) DAAL_C11_OVERRIDE
    {
        DAAL_CHECK(!baseRng.loadState((const void *)src), ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

    services::Status leapfrogImpl(size_t threadNum, size_t nThreads) DAAL_C11_OVERRIDE
    {
        int errcode = baseRng.leapfrog(threadNum, nThreads);
        services::Status s;
        if(errcode == leapfrogMethodErrcode) s.add(ErrorLeapfrogUnsupported);
        else if(errcode) s.add(ErrorIncorrectErrorcodeFromGenerator);
        return s;
    }

    services::Status skipAheadImpl(size_t nSkip) DAAL_C11_OVERRIDE
    {
        int errcode = baseRng.skipAhead(nSkip);
        services::Status s;
        if(errcode == skipAheadMethodErrcode) s.add(ErrorSkipAheadUnsupported);
        else if (errcode) s.add(ErrorIncorrectErrorcodeFromGenerator);
        return s;
    }

    virtual BatchImpl<cpu, algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new BatchImpl<cpu, algorithmFPType, method>(*this);
    }

    bool hasSupport(engines::internal::ParallelizationTechnique technique) const DAAL_C11_OVERRIDE
    {
        switch(technique)
        {
            case engines::internal::family: return false;
            case engines::internal::skipahead: return true;
            case engines::internal::leapfrog: return false;
        }
        return false;
    }

    ~BatchImpl() {}

protected:
    BatchImpl(const BatchImpl<cpu, algorithmFPType, method> &other) : super1(other), super2(other), baseRng(other.baseRng) {}
    daal::internal::BaseRNGs<cpu> baseRng;
};

} // namespace interface1
} // namespace mt19937
} // namespace engines
} // namespace algorithms
} // namespace daal
