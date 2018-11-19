/* file: initializers_impl.i */
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
