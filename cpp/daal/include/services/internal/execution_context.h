/* file: execution_context.h */
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

#ifndef __DAAL_SERVICES_EXECUTION_CONTEXT_H__
#define __DAAL_SERVICES_EXECUTION_CONTEXT_H__

#include "services/internal/utilities.h"
#include "services/internal/sycl/execution_context.h"

namespace daal
{
namespace services
{
namespace internal
{
namespace interface1
{
/**
 * @defgroup sycl SYCL*
 * \brief Contains classes designed to work with SYCL* and call
 * oneAPI implementations of algorithms
 * @{
 */

/**
 *  <a name="DAAL-CLASS-SERVICES__EXECUTIONCONTEXT"></a>
 *  \brief Base class for device information needed to perform
 *   computations
 */
class ExecutionContext : public Base
{
    friend class daal::services::internal::ImplAccessor;

private:
    typedef daal::services::internal::sycl::ExecutionContextIface ImplType;

public:
    ExecutionContext() {}

protected:
    explicit ExecutionContext(ImplType * impl) : _impl(impl) {}
    explicit ExecutionContext(ImplType * impl, bool needEmptyDeleter)
    {
        // This branch is needed to avoid problems with deleting SYCL entities
        // after SYCL RT static objects are already released.
        // This is caused by "C++ static initialization order fiasco" problem between
        // Intel(R) oneAPI Data Analytics Library (oneDAL) static Environment object and internal static contexts of SYCL RT.
        // Here we solve this temporary with a small memory leak.
        // TODO: remove this after complete transition to DPC++ kernels.
        if (needEmptyDeleter)
        {
            _impl = SharedPtr<ImplType>(impl, EmptyDeleter());
        }
        else
        {
            _impl = SharedPtr<ImplType>(impl);
        }
    }

    const SharedPtr<ImplType> & getImplPtr() const { return _impl; }

private:
    SharedPtr<ImplType> _impl;
};

/**
 *  <a name="DAAL-CLASS-SERVICES__CPUEXECUTIONCONTEXT"></a>
 *  \brief Implementation of a CPU-host context class
 */
class CpuExecutionContext : public ExecutionContext
{
private:
    typedef services::internal::sycl::CpuExecutionContextImpl ImplType;

public:
    CpuExecutionContext() : ExecutionContext(new ImplType()) {}
};
/** @} */
} // namespace interface1

using interface1::ExecutionContext;
using interface1::CpuExecutionContext;

} // namespace internal
} // namespace services
} // namespace daal

#ifdef DAAL_SYCL_INTERFACE
    #include "services/internal/sycl/execution_context_sycl.h"

namespace daal
{
namespace services
{
namespace internal
{
namespace interface1
{
/** @ingroup sycl
 * @{
 */

/**
 *  <a name="DAAL-CLASS-SERVICES__SYCLEXECUTIONCONTEXT"></a>
 *  \brief Implementation of a device context class
 *  based on SYCL* queue object
 */
class SyclExecutionContext : public ExecutionContext
{
public:
    /** Constructor from SYCL* queue.
     *  When this execution context is selected, all computations
     *  are performed on the device associated with the queue
     *  \param[in] deviceQueue SYCL* queue object to the device that is selected to perform computations
     */
    SyclExecutionContext(const ::sycl::queue & deviceQueue, const bool fromPython = false)
        : ExecutionContext(createContext(deviceQueue, fromPython), !deviceQueue.get_device().is_cpu())
    {}

private:
    static daal::services::internal::sycl::ExecutionContextIface * createContext(const ::sycl::queue & queue, const bool fromPython = false)
    {
        /* XXX: Workaround to fix performance on CPU: SYCL* runtime loads one
                thread with active spin-lock that waits for submissions in a queue.
                In CPU mode DAAL does not submit kernels, and runs CPU code via TBB.
                Spin-lock is active while the queue persists. We do not persist
                the queue and avoid running spin-lock in a queue while any DAAL
                algorithm is running. */
        if (queue.get_device().is_cpu())
        {
            return new daal::services::internal::sycl::CpuExecutionContextImpl();
        }
        else
        {
            try
            {
                return new daal::services::internal::sycl::SyclExecutionContextImpl(queue, fromPython);
            }
            catch (const std::runtime_error & e)
            {
                throw e;
            }
        }
    }
};
/** @} */
} // namespace interface1

using interface1::SyclExecutionContext;

} // namespace internal
} // namespace services
} // namespace daal
#endif // DAAL_SYCL_INTERFACE

namespace daal
{
namespace services
{
namespace internal
{
namespace interface1
{
DAAL_EXPORT sycl::ExecutionContextIface & getDefaultContext();

} // namespace interface1

using interface1::getDefaultContext;

} // namespace internal
} // namespace services
} // namespace daal

#endif
