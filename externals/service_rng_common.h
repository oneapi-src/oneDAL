/* file: service_rng_common.h */
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

/*
//++
//  Common RNG classes.
//--
*/

#ifndef __SERVICE_RNG_COMMON_H__
#define __SERVICE_RNG_COMMON_H__

namespace daal
{
namespace internal
{
template <CpuType cpu>
class BaseRNGIface
{
public:
    virtual int getStateSize() const                        = 0;
    virtual int saveState(void * dest) const                = 0;
    virtual int loadState(const void * src)                 = 0;
    virtual int leapfrog(size_t threadNum, size_t nThreads) = 0;
    virtual int skipAhead(size_t nSkip)                     = 0;
    virtual void * getState()                               = 0;
};

} // namespace internal
} // namespace daal

#endif
