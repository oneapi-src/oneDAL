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

DAAL_EXPORT sycl::ExecutionContextIface & getDefaultContext();

} // namespace interface1

using interface1::ExecutionContext;
using interface1::CpuExecutionContext;
using interface1::getDefaultContext;

} // namespace internal
} // namespace services
} // namespace daal

#endif
