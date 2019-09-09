/* file: initializers_impl.i */
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
//  Implementation of useful functions to be used by initializers
//--
*/

#ifndef __INITIALIZERS_IMPL_I__
#define __INITIALIZERS_IMPL_I__

#include "mt19937_batch_impl.h"

#include "service_utils.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace internal
{

template<CpuType cpu>
class EngineImpl
{
private:
    bool _engineAllocatedOnOurSide;
    engines::internal::BatchBaseImpl *_engine;

public:
    EngineImpl(engines::BatchBase *engine)
    {
        _engineAllocatedOnOurSide = false;
        _engine = dynamic_cast<engines::internal::BatchBaseImpl *>(engine);

        if (!engine)
        {
            _engineAllocatedOnOurSide = true;
            _engine = new engines::mt19937::internal::BatchImpl<cpu>();
        }
    }

    ~EngineImpl()
    {
        if (_engineAllocatedOnOurSide)
        {
            delete _engine;
        }
    }

    EngineImpl(const EngineImpl &) = delete;
    EngineImpl &operator=(const EngineImpl &) = delete;

    inline engines::internal::BatchBaseImpl *get() { return _engine; }
    inline engines::internal::BatchBaseImpl &operator * () { return *_engine; }
};

} // internal
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
