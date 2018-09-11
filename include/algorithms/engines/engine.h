/* file: engine.h */
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
    typedef algorithms::engines::Input  InputType;
    typedef algorithms::engines::Result ResultType;

    InputType  input;   /*!< Input of the engine */

    BatchBase() {}
    virtual ~BatchBase() {}

    /**
     * Saves current engine state to destination
     * \param[in] dest  Destination to save the state
     *
     * \return Status of computations
     */
    services::Status saveState(byte* dest) const
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

    /**
     * Enables the usage of current engine in parallel regions of code with leapfrog method
     * \param[in] threadIdx  Index of the thread
     * \param[in] nThreads   Number of threads
     *
     * \return Status of computations
     */
    services::Status leapfrog(size_t threadIdx, size_t nThreads)
    {
        return leapfrogImpl(threadIdx, nThreads);
    }

    /**
     * Enables the usage of current engine in parallel regions of code with skipAhead method
     * \param[in] nSkip  Number of elements that will be skipped
     *
     * \return Status of computations
     */
    services::Status skipAhead(size_t nSkip)
    {
        return skipAheadImpl(nSkip);
    }

    /**
     * Returns a pointer to the newly allocated engine
     * with a copy of input objects and parameters of this engine
     * \return Pointer to the newly allocated engine
     */
    services::SharedPtr<BatchBase> clone() const
    {
        return services::SharedPtr<BatchBase>(cloneImpl());
    }

protected:
    virtual services::Status saveStateImpl(byte* dest) const { return services::Status(); }
    virtual services::Status loadStateImpl(const byte* src) { return services::Status(); }
    virtual services::Status leapfrogImpl(size_t threadNum, size_t nThreads) { return services::Status(services::ErrorMethodNotSupported); }
    virtual services::Status skipAheadImpl(size_t nSkip) { return services::Status(); }
    virtual BatchBase *cloneImpl() const = 0;
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
