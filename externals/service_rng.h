/* file: service_rng.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Template wrappers for RNG functions.
//--
*/


#ifndef __SERVICE_RNG_H__
#define __SERVICE_RNG_H__

#include "daal_defines.h"
#include "service_memory.h"

#include "service_rng_mkl.h"

namespace daal
{
namespace internal
{

/*
// Template functions definition
*/
template<typename baseType, CpuType cpu, template<typename, CpuType> class _impl=mkl::MklIntRng>
struct IntRng
{
    typedef typename _impl<baseType,cpu>::SizeType SizeType;

    IntRng(int seed = 777) : rng(seed) {}

    void uniform(SizeType n, baseType a, baseType b, baseType* r)
    {
        rng.uniform(n, a, b, r);
    }

    void bernoulli(SizeType n, baseType* r, double p)
    {
        rng.bernoulli(n, r, p);
    }

    void uniformWithoutReplacement(SizeType n, baseType a, baseType b, baseType* r)
    {
        for (int i = 0; i < n; i++)
        {
            uniform(1, a + i, b, r + i);
        }
        for (int i = 0; i < n; i++)
        {
            int shift = 0;
            for (int j = 0; j < i; j++)
            {
                shift += (r[i] <= r[j]);
            }
            r[i] -= shift;
        }
    }

    int getStreamSize() const
    {
        return rng.getStreamSize();
    }

    void saveStream(void* dest) const
    {
        rng.saveStream(dest);
    }

    void loadStream(const void* src)
    {
        rng.loadStream(src);
    }

private:
    _impl<baseType,cpu> rng;
};

template<typename baseType, CpuType cpu, template<typename, CpuType> class _impl=mkl::MklUniformRng >
struct UniformRng
{
    typedef typename _impl<baseType,cpu>::SizeType SizeType;

    UniformRng(int seed = 777) : rng(seed) {}

    void uniform(SizeType n, baseType a, baseType b, baseType* r)
    {
        rng.uniform(n, a, b, r);
    }

    int getStreamSize() const
    {
        return rng.getStreamSize();
    }

    void saveStream(void* dest) const
    {
        rng.saveStream(dest);
    }

    void loadStream(const void* src)
    {
        rng.loadStream(src);
    }

private:
    _impl<baseType,cpu> rng;
};

} // namespace internal
} // namespace daal

#endif
