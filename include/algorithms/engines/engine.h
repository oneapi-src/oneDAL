/* file: engine.h */
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
//  Implementation engine.
//--
*/

#ifndef __ENGINE_H__
#define __ENGINE_H__

#include "algorithms/engines/engine_types.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
/**
 * @ingroup engines
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__BATCHBASE"></a>
 *  \brief Class representing an engine
 */
class DAAL_EXPORT BatchBase : public daal::algorithms::Analysis<batch>
{
public:
    Input  input;   /*!< Input of the engine */

    BatchBase() {}
    virtual ~BatchBase() {}

    /**
     * Saves current engine state to destination
     * \param[in] dest  Destination to save the state
     *
     * \return Status of computations
     */
    services::Status saveState(byte* dest)
    {
        return saveStateImpl(dest);
    }

    /**
     * Rewrites current state with source one
     * \param[in] src  Source state
     *
     * \return Status of computations
     */
    services::Status loadState(const byte* src)
    {
        return loadStateImpl(src);
    }
protected:
    virtual services::Status saveStateImpl(byte* dest) { return services::Status(); }
    virtual services::Status loadStateImpl(const byte* src) { return services::Status(); }
};
typedef services::SharedPtr<BatchBase> EnginePtr;

} // namespace interface1
using interface1::BatchBase;
using interface1::EnginePtr;
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal
#endif
