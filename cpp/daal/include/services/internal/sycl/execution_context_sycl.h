/* file: execution_context_sycl.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_EXECUTION_CONTEXT_SYCL_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_EXECUTION_CONTEXT_SYCL_H__

#ifndef DAAL_SYCL_INTERFACE
    #error "DAAL_SYCL_INTERFACE must be defined to include this file"
#endif

#include <CL/cl.h>
#include <sycl/sycl.hpp>
#include <backend/opencl.hpp>

#include "services/daal_string.h"
#include "services/internal/hash_table.h"
#include "services/internal/sycl/execution_context.h"
#include "services/internal/sycl/kernel_scheduler_sycl.h"
#include "services/internal/sycl/error_handling_sycl.h"
#include "services/internal/sycl/math/blas_executor.h"
#include "services/internal/sycl/math/lapack_executor.h"

/// \cond INTERNAL
namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
namespace interface1
{
class OpenClKernelFactory : public Base, public ClKernelFactoryIface
{
public:
    explicit OpenClKernelFactory(::sycl::queue & deviceQueue)
        : _currentProgramRef(nullptr), _executionTarget(ExecutionTargetIds::unspecified), _deviceQueue(deviceQueue)
    {}

    void build(ExecutionTargetId target, const char * name, const char * program, const char * options, Status & status) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(name);
        DAAL_ASSERT(program);

        String key = name;
        DAAL_CHECK_COND_ERROR(key.c_str(), status, ErrorMemoryAllocationFailed);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        const bool res = programHashTable.contain(key, status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        if (!res)
        {
#ifndef DAAL_DISABLE_LEVEL_ZERO
            const bool isOpenCLBackendAvailable = !_deviceQueue.get_device().template get_info< ::sycl::info::device::opencl_c_version>().empty();
            if (isOpenCLBackendAvailable)
            {
#endif // DAAL_DISABLE_LEVEL_ZERO

                // OpenCl branch
                auto programPtr =
                    OpenClProgramRef::create(::sycl::get_native< ::sycl::backend::opencl>(_deviceQueue.get_context()),
                                             ::sycl::get_native< ::sycl::backend::opencl>(_deviceQueue.get_device()), name, program, options, status);
                DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

                programHashTable.add(key, programPtr, status);
                DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

                _currentProgramRef = programPtr.get();

#ifndef DAAL_DISABLE_LEVEL_ZERO
            }
            else
            {
                // Level zero branch
                if (nullptr == _levelZeroOpenClInteropContext.getOpenClDeviceRef().get())
                {
                    _levelZeroOpenClInteropContext.reset(_deviceQueue, status);
                    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
                }

                auto programPtr =
                    OpenClProgramRef::create(_levelZeroOpenClInteropContext.getOpenClContextRef().get(),
                                             _levelZeroOpenClInteropContext.getOpenClDeviceRef().get(), _deviceQueue, name, program, options, status);
                DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

                programHashTable.add(key, programPtr, status);
                DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

                _currentProgramRef = programPtr.get();
            }
#endif // DAAL_DISABLE_LEVEL_ZERO
        }
        else
        {
            _currentProgramRef = programHashTable.get(key, status).get();
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
        }

        _executionTarget = target;
    }

    KernelPtr getKernel(const char * kernelName, Status & status) DAAL_C11_OVERRIDE
    {
        if (!_currentProgramRef)
        {
            status |= ErrorExecutionContext;
            return KernelPtr();
        }

        String kernelNameStr = kernelName;
        DAAL_CHECK_COND_ERROR(kernelNameStr.c_str(), status, ErrorMemoryAllocationFailed);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());

        String key = _currentProgramRef->getName();
        DAAL_CHECK_COND_ERROR(key.c_str(), status, ErrorMemoryAllocationFailed);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());

        key.add(kernelNameStr);
        DAAL_CHECK_COND_ERROR(key.c_str(), status, ErrorMemoryAllocationFailed);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());

        bool res = kernelHashTable.contain(key, status);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());

        if (res)
        {
            auto kernel = kernelHashTable.get(key, status);
            DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());

            return kernel;
        }
        else
        {
            KernelPtr kernel;
#ifndef DAAL_DISABLE_LEVEL_ZERO
            const bool isOpenCLBackendAvailable = !_deviceQueue.get_device().template get_info< ::sycl::info::device::opencl_c_version>().empty();
            if (isOpenCLBackendAvailable)
            {
#endif // DAAL_DISABLE_LEVEL_ZERO

                // OpenCL branch
                auto kernelRef = OpenClKernelRef(_currentProgramRef->get(), kernelNameStr, status);
                DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());

                kernel = OpenClKernelNative::create(_executionTarget, *_currentProgramRef, kernelRef, status);
                DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());

#ifndef DAAL_DISABLE_LEVEL_ZERO
            }
            else
            {
                // Level zero branch
                auto kernelRef = OpenClKernelLevelZeroRef(*_currentProgramRef, kernelNameStr, status);
                DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());

                kernel = OpenClKernelLevelZero::create(_executionTarget, *_currentProgramRef, kernelRef, status);
                DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());
            }
#endif // DAAL_DISABLE_LEVEL_ZERO
            kernelHashTable.add(key, kernel, status);
            DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, KernelPtr());

            return kernel;
        }
    }

private:
    static const size_t SIZE_HASHTABLE_PROGRAM = 1024;
    static const size_t SIZE_HASHTABLE_KERNEL  = 4096;
    HashTable<OpenClProgramRef, SIZE_HASHTABLE_PROGRAM> programHashTable;
    HashTable<KernelIface, SIZE_HASHTABLE_KERNEL> kernelHashTable;

    OpenClProgramRef * _currentProgramRef;
#ifndef DAAL_DISABLE_LEVEL_ZERO
    LevelZeroOpenClInteropContext _levelZeroOpenClInteropContext;
#endif // DAAL_DISABLE_LEVEL_ZERO

    ExecutionTargetId _executionTarget;
    ::sycl::queue & _deviceQueue;
};

class [[deprecated("CPP SYCL interfaces have been removed as of 2024.0 release.")]] SyclExecutionContextImpl : public Base,
                                                                                                               public ExecutionContextIface
{
public:
    explicit SyclExecutionContextImpl(const ::sycl::queue & deviceQueue, const bool fromPython = false)
        : _deviceQueue(deviceQueue), _kernelFactory(_deviceQueue), _kernelScheduler(_deviceQueue)
    {
        if (!fromPython)
        {
            throw std::runtime_error("CPP SYCL interfaces have been removed as of 2024.0 release.");
        }
        const auto & device          = _deviceQueue.get_device();
        _infoDevice.isCpu            = device.is_cpu();
        _infoDevice.maxWorkGroupSize = device.get_info< ::sycl::info::device::max_work_group_size>();
        _infoDevice.maxMemAllocSize  = device.get_info< ::sycl::info::device::max_mem_alloc_size>();
        _infoDevice.globalMemSize    = device.get_info< ::sycl::info::device::global_mem_size>();
    }

    void run(const KernelRange & range, const KernelPtr & kernel, const KernelArguments & args, Status & status) DAAL_C11_OVERRIDE
    {
        kernel->schedule(_kernelScheduler, range, args, status);
    }

    void run(const KernelNDRange & range, const KernelPtr & kernel, const KernelArguments & args, Status & status) DAAL_C11_OVERRIDE
    {
        kernel->schedule(_kernelScheduler, range, args, status);
    }

    void gemm(math::Transpose transa, math::Transpose transb, size_t m, size_t n, size_t k, double alpha, const UniversalBuffer & a_buffer,
              size_t lda, size_t offsetA, const UniversalBuffer & b_buffer, size_t ldb, size_t offsetB, double beta, UniversalBuffer & c_buffer,
              size_t ldc, size_t offsetC, Status & status) DAAL_C11_OVERRIDE
    {
        math::GemmExecutor::run(_deviceQueue, transa, transb, m, n, k, alpha, a_buffer, lda, offsetA, b_buffer, ldb, offsetB, beta, c_buffer, ldc,
                                offsetC, status);
    }

    void syrk(math::UpLo upper_lower, math::Transpose trans, size_t n, size_t k, double alpha, const UniversalBuffer & a_buffer, size_t lda,
              size_t offsetA, double beta, UniversalBuffer & c_buffer, size_t ldc, size_t offsetC, Status & status) DAAL_C11_OVERRIDE
    {
        math::SyrkExecutor::run(_deviceQueue, upper_lower, trans, n, k, alpha, a_buffer, lda, offsetA, beta, c_buffer, ldc, offsetC, status);
    }

    void axpy(const uint32_t n, const double a, const UniversalBuffer x_buffer, const int incx, const UniversalBuffer y_buffer, const int incy,
              Status & status) DAAL_C11_OVERRIDE
    {
        math::AxpyExecutor::run(_deviceQueue, n, a, x_buffer, incx, y_buffer, incy, status);
    }

    void potrf(math::UpLo uplo, size_t n, UniversalBuffer & a_buffer, size_t lda, Status & status) DAAL_C11_OVERRIDE
    {
        math::PotrfExecutor::run(_deviceQueue, uplo, n, a_buffer, lda, status);
    }

    void potrs(math::UpLo uplo, size_t n, size_t ny, UniversalBuffer & a_buffer, size_t lda, UniversalBuffer & b_buffer, size_t ldb, Status & status)
        DAAL_C11_OVERRIDE
    {
        math::PotrsExecutor::run(_deviceQueue, uplo, n, ny, a_buffer, lda, b_buffer, ldb, status);
    }

    UniversalBuffer allocate(TypeId type, size_t bufferSize, Status & status) DAAL_C11_OVERRIDE
    {
        return BufferAllocator::allocate(_deviceQueue, type, bufferSize, status);
    }

    void copy(UniversalBuffer dest, size_t desOffset, UniversalBuffer src, size_t srcOffset, size_t count, Status & status) DAAL_C11_OVERRIDE
    {
        BufferCopier::copy(_deviceQueue, dest, desOffset, src, srcOffset, count, status);
    }

    void copy(UniversalBuffer dest, size_t desOffset, void * src, size_t srcCount, size_t srcOffset, size_t count, Status & status) DAAL_C11_OVERRIDE
    {
        ArrayCopier::copy(_deviceQueue, dest, desOffset, src, srcCount, srcOffset, count, status);
    }

    void fill(UniversalBuffer dest, double value, Status & status) DAAL_C11_OVERRIDE
    {
        BufferFiller::fill(_deviceQueue, dest, value, status);
    }

    ClKernelFactoryIface & getClKernelFactory() DAAL_C11_OVERRIDE
    {
        return _kernelFactory;
    }

    InfoDevice & getInfoDevice() DAAL_C11_OVERRIDE
    {
        return _infoDevice;
    }

    const ::sycl::queue & getQueue() const
    {
        return _deviceQueue;
    }

private:
    ::sycl::queue _deviceQueue;
    OpenClKernelFactory _kernelFactory;
    SyclKernelScheduler _kernelScheduler;
    InfoDevice _infoDevice;
};
} // namespace interface1

using interface1::SyclExecutionContextImpl;

} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
/// \endcond

#endif
