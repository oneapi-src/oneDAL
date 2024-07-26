/* file: engine_batch_impl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the class defining the engine
//--
*/

#ifndef __ENGINE_BATCH_IMPL_H__
#define __ENGINE_BATCH_IMPL_H__

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace internal
{
enum ParallelizationTechnique
{
    skipahead = 1,
    leapfrog  = 2,
    family    = 4
};

class BatchBaseImpl
{
public:
    BatchBaseImpl(size_t seed) : _seed(seed) {}
    size_t getSeed() const { return _seed; }
    virtual void * getState()                = 0;
    virtual int skipAheadoneDAL(size_t skip) = 0;
    virtual int getStateSize() const         = 0;
    virtual ~BatchBaseImpl() {}
    virtual bool hasSupport(ParallelizationTechnique technique) const = 0;

protected:
    const size_t _seed;
};

} // namespace internal
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
