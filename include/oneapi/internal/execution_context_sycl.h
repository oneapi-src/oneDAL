/* file: execution_context_sycl.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#ifdef DAAL_SYCL_INTERFACE
    #ifndef __DAAL_ONEAPI_INTERNAL_EXECUTION_CONTEXT_SYCL_H__
        #define __DAAL_ONEAPI_INTERNAL_EXECUTION_CONTEXT_SYCL_H__

        #include "services/daal_string.h"
        #include "services/internal/hash_table.h"
        #include "oneapi/internal/execution_context.h"
        #include "oneapi/internal/kernel_scheduler_sycl.h"
        #include "oneapi/internal/math/blas_executor.h"
        #include "oneapi/internal/math/lapack_executor.h"
        #include "oneapi/internal/error_handling.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace interface1
{
class OpenClKernelFactory : public Base, public ClKernelFactoryIface
{
public:
    explicit OpenClKernelFactory(cl::sycl::queue & deviceQueue)
        : _currentProgramRef(nullptr), _executionTarget(ExecutionTargetIds::unspecified), _deviceQueue(deviceQueue)
    {}

    ~OpenClKernelFactory() DAAL_C11_OVERRIDE {}

    void build(ExecutionTargetId target, const char * name, const char * program, const char * options = "",
               services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        services::Status localStatus;
        services::String key = name;
        const bool res       = programHashTable.contain(key, localStatus);
        if (!localStatus.ok())
        {
            services::internal::tryAssignStatus(status, localStatus);
            return;
        }
        if (!res)
        {
        #ifndef DAAL_DISABLE_LEVEL_ZERO
            const bool isOpenCLBackendAvailable = !_deviceQueue.get_device().template get_info<sycl::info::device::opencl_c_version>().empty();
            if (isOpenCLBackendAvailable)
            {
        #endif // DAAL_DISABLE_LEVEL_ZERO \
            // OpenCl branch
                auto programPtr = services::SharedPtr<OpenClProgramRef>(
                    new OpenClProgramRef(_deviceQueue.get_context().get(), _deviceQueue.get_device().get(), name, program, options, &localStatus));

                if (!localStatus.ok())
                {
                    services::internal::tryAssignStatus(status, localStatus);
                    return;
                }
                programHashTable.add(key, programPtr, localStatus);
                if (!localStatus.ok())
                {
                    services::internal::tryAssignStatus(status, localStatus);
                    return;
                }

                _currentProgramRef = programPtr.get();
        #ifndef DAAL_DISABLE_LEVEL_ZERO
            }
            else
            {
                // Level zero branch
                if (nullptr == _levelZeroOpenClInteropContext.getOpenClDeviceRef().get())
                {
                    _levelZeroOpenClInteropContext.reset(_deviceQueue, &localStatus);
                }
                if (!localStatus.ok())
                {
                    services::internal::tryAssignStatus(status, localStatus);
                    return;
                }

                auto programPtr = services::SharedPtr<OpenClProgramRef>(new OpenClProgramRef(
                    _levelZeroOpenClInteropContext.getOpenClContextRef().get(), _levelZeroOpenClInteropContext.getOpenClDeviceRef().get(),
                    _deviceQueue, name, program, options, &localStatus));
                if (!localStatus.ok())
                {
                    services::internal::tryAssignStatus(status, localStatus);
                    return;
                }
                programHashTable.add(key, programPtr, localStatus);
                if (!localStatus.ok())
                {
                    services::internal::tryAssignStatus(status, localStatus);
                    return;
                }

                _currentProgramRef = programPtr.get();
            }
        #endif // DAAL_DISABLE_LEVEL_ZERO
        }
        else
        {
            _currentProgramRef = programHashTable.get(key, localStatus).get();
            if (!localStatus.ok())
            {
                services::internal::tryAssignStatus(status, localStatus);
                return;
            }
        }

        _executionTarget = target;
    }

    KernelPtr getKernel(const char * kernelName, services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        if (_currentProgramRef == nullptr)
        {
            services::internal::tryAssignStatus(status, services::ErrorExecutionContext);
            return KernelPtr();
        }

        services::Status localStatus;

        services::String key = _currentProgramRef->getName();
        key.add(kernelName);

        bool res = kernelHashTable.contain(key, localStatus);
        if (!localStatus.ok())
        {
            services::internal::tryAssignStatus(status, localStatus);
            return KernelPtr();
        }
        if (res)
        {
            auto kernel = kernelHashTable.get(key, localStatus);
            services::internal::tryAssignStatus(status, localStatus);
            return kernel;
        }
        else
        {
            KernelPtr kernel;
        #ifndef DAAL_DISABLE_LEVEL_ZERO
            const bool isOpenCLBackendAvailable = !_deviceQueue.get_device().template get_info<sycl::info::device::opencl_c_version>().empty();
            if (isOpenCLBackendAvailable)
            {
        #endif // DAAL_DISABLE_LEVEL_ZERO \
            // OpenCl branch
                auto kernelRef = OpenClKernelRef(_currentProgramRef->get(), kernelName, &localStatus);
                if (!localStatus.ok())
                {
                    services::internal::tryAssignStatus(status, localStatus);
                    return KernelPtr();
                }
                kernel.reset(new OpenClKernelNative(_executionTarget, *_currentProgramRef, kernelRef));
        #ifndef DAAL_DISABLE_LEVEL_ZERO
            }
            else
            {
                // Level zero branch
                auto kernelRef = OpenClKernelLevelZeroRef(kernelName, &localStatus);

                if (!localStatus.ok())
                {
                    services::internal::tryAssignStatus(status, localStatus);
                    return KernelPtr();
                }
                kernel.reset(new OpenClKernelLevelZero(_executionTarget, *_currentProgramRef, kernelRef));
            }
        #endif // DAAL_DISABLE_LEVEL_ZERO
            kernelHashTable.add(key, kernel, localStatus);
            if (!localStatus.ok())
            {
                services::internal::tryAssignStatus(status, localStatus);
                return KernelPtr();
            }
            return kernel;
        }
    }

private:
    static const size_t SIZE_HASHTABLE_PROGRAM = 1024;
    static const size_t SIZE_HASHTABLE_KERNEL  = 4096;
    services::internal::HashTable<OpenClProgramRef, SIZE_HASHTABLE_PROGRAM> programHashTable;
    services::internal::HashTable<KernelIface, SIZE_HASHTABLE_KERNEL> kernelHashTable;

    OpenClProgramRef * _currentProgramRef;
        #ifndef DAAL_DISABLE_LEVEL_ZERO
    LevelZeroOpenClInteropContext _levelZeroOpenClInteropContext;
        #endif // DAAL_DISABLE_LEVEL_ZERO

    ExecutionTargetId _executionTarget;
    cl::sycl::queue & _deviceQueue;
};

class SyclExecutionContextImpl : public Base, public ExecutionContextIface
{
public:
    explicit SyclExecutionContextImpl(const cl::sycl::queue & deviceQueue)
        : _deviceQueue(deviceQueue), _kernelFactory(_deviceQueue), _kernelScheduler(_deviceQueue)
    {
        const auto & device          = _deviceQueue.get_device();
        _infoDevice.isCpu            = device.is_cpu() || device.is_host();
        _infoDevice.maxWorkGroupSize = device.get_info<cl::sycl::info::device::max_work_group_size>();
        _infoDevice.maxNumSubGroups  = device.is_host() ? 16 : device.get_info<cl::sycl::info::device::max_num_sub_groups>();
    }

    void run(const KernelRange & range, const KernelPtr & kernel, const KernelArguments & args, services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        // TODO: Check for input arguments
        // TODO: Need to save reference to kernel to prevent
        //       releasing in case of asynchronous execution?
        kernel->schedule(_kernelScheduler, range, args, status);
    }

    void run(const KernelNDRange & range, const KernelPtr & kernel, const KernelArguments & args,
             services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        // TODO: Check for input arguments
        // TODO: Need to save reference to kernel to prevent
        //       releasing in case of asynchronous execution?
        kernel->schedule(_kernelScheduler, range, args, status);
    }

    void gemm(math::Transpose transa, math::Transpose transb, size_t m, size_t n, size_t k, double alpha, const UniversalBuffer & a_buffer,
              size_t lda, size_t offsetA, const UniversalBuffer & b_buffer, size_t ldb, size_t offsetB, double beta, UniversalBuffer & c_buffer,
              size_t ldc, size_t offsetC, services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(a_buffer.type() == b_buffer.type());
        DAAL_ASSERT(b_buffer.type() == c_buffer.type());

        // TODO: Check for input arguments
        math::GemmExecutor::run(_deviceQueue, transa, transb, m, n, k, alpha, a_buffer, lda, offsetA, b_buffer, ldb, offsetB, beta, c_buffer, ldc,
                                offsetC, status);
    }

    void syrk(math::UpLo upper_lower, math::Transpose trans, size_t n, size_t k, double alpha, const UniversalBuffer & a_buffer, size_t lda,
              size_t offsetA, double beta, UniversalBuffer & c_buffer, size_t ldc, size_t offsetC,
              services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(a_buffer.type() == c_buffer.type());

        math::SyrkExecutor::run(_deviceQueue, upper_lower, trans, n, k, alpha, a_buffer, lda, offsetA, beta, c_buffer, ldc, offsetC, status);
    }

    void axpy(const uint32_t n, const double a, const UniversalBuffer x_buffer, const int incx, const UniversalBuffer y_buffer, const int incy,
              services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(x_buffer.type() == y_buffer.type());

        math::AxpyExecutor::run(_deviceQueue, n, a, x_buffer, incx, y_buffer, incy, status);
    }

    void potrf(math::UpLo uplo, size_t n, UniversalBuffer & a_buffer, size_t lda, services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        math::PotrfExecutor::run(_deviceQueue, uplo, n, a_buffer, lda, status);
    }

    void potrs(math::UpLo uplo, size_t n, size_t ny, UniversalBuffer & a_buffer, size_t lda, UniversalBuffer & b_buffer, size_t ldb,
               services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(a_buffer.type() == b_buffer.type());
        math::PotrsExecutor::run(_deviceQueue, uplo, n, ny, a_buffer, lda, b_buffer, ldb, status);
    }

    UniversalBuffer allocate(TypeId type, size_t bufferSize, services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        try
        {
            auto buffer = BufferAllocator::allocate(type, bufferSize);
            return buffer;
        }
        catch (cl::sycl::exception const & e)
        {
            convertSyclExceptionToStatus(e, status);
            return UniversalBuffer();
        }
    }

    void copy(UniversalBuffer dest, size_t desOffset, UniversalBuffer src, size_t srcOffset, size_t count,
              services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(dest.type() == src.type());
        // TODO: Thread safe?
        try
        {
            BufferCopier::copy(_deviceQueue, dest, desOffset, src, srcOffset, count);
        }
        catch (cl::sycl::exception const & e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }

    void fill(UniversalBuffer dest, double value, services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        try
        {
            BufferFiller::fill(_deviceQueue, dest, value);
        }
        catch (cl::sycl::exception const & e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }

    ClKernelFactoryIface & getClKernelFactory() DAAL_C11_OVERRIDE { return _kernelFactory; }

    InfoDevice & getInfoDevice() DAAL_C11_OVERRIDE { return _infoDevice; }

    void copy(UniversalBuffer dest, size_t desOffset, void * src, size_t srcOffset, size_t count,
              services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        try
        {
            ArrayCopier::copy(_deviceQueue, dest, desOffset, src, srcOffset, count);
        }
        catch (cl::sycl::exception const & e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }

private:
    cl::sycl::queue _deviceQueue;
    OpenClKernelFactory _kernelFactory;
    SyclKernelScheduler _kernelScheduler;
    InfoDevice _infoDevice;
};

/** } */
} // namespace interface1

using interface1::SyclExecutionContextImpl;

} // namespace internal
} // namespace oneapi
} // namespace daal

    #endif
#endif // DAAL_SYCL_INTERFACE
