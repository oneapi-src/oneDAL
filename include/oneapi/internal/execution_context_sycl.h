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

        #include <vector>
        #include <cstring>
        #include <CL/cl.h>
        #include <CL/sycl.hpp>

        #include "services/daal_string.h"
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
private:
    class ProgramCacheEntry
    {
    private:
        OpenClProgramRef * _program;

    public:
        ProgramCacheEntry() : _program(nullptr) {}

        ~ProgramCacheEntry() { delete _program; }

        void setProgram(OpenClProgramRef * program) { _program = program; }

        OpenClProgramRef * getProgram() { return _program; }

        const char * getName(services::Status * status = nullptr)
        {
            if (!_program)
            {
                if (status)
                {
                    status->add(services::ErrorExecutionContext);
                }
                return nullptr;
            }
            return _program->getName();
        }
    };

    class KernelCacheEntry
    {
    private:
        KernelPtr _kernel;
        services::String _name;

    public:
        KernelCacheEntry() : _kernel(), _name() {}

        ~KernelCacheEntry() {}

        void setKernel(KernelPtr kernel, const char * name)
        {
            _name   = name;
            _kernel = kernel;
        }

        KernelPtr getKernel() { return _kernel; }

        const char * getName() { return _name.c_str(); }
    };

public:
    explicit OpenClKernelFactory(cl::sycl::queue & deviceQueue)
        : _clProgramRef(nullptr), _executionTarget(ExecutionTargetIds::unspecified), _deviceQueue(deviceQueue)
    {}

    void build(ExecutionTargetId target, const char * key, const char * program, const char * options = "",
               services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        // TODO Rework of "cache"

        uint64_t id = hash(key) % SIZE_CACHE_PROGRAM;

        while (_clProgramCache[id].getProgram() != nullptr && strcmp(_clProgramCache[id].getName(), key) != 0)
        {
            id = (id + 1) % SIZE_CACHE_PROGRAM;
        }

        if (_clProgramCache[id].getProgram())
        {
            _clProgramRef    = _clProgramCache[id].getProgram();
            _executionTarget = target;
        }
        else
        {
            _clProgramCache[id].setProgram(
                new OpenClProgramRef(_deviceQueue.get_context().get(), _deviceQueue.get_device().get(), key, program, options, status));
            if (status != nullptr && !status->ok())
            {
                return;
            }
            _clProgramRef    = _clProgramCache[id].getProgram();
            _executionTarget = target;
        }
    }

    KernelPtr getKernel(const char * kernelName, services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(_clProgramRef);
        DAAL_ASSERT(*_clProgramRef);
        KernelPtr kernelPtr;

        services::String keyCache = _clProgramRef->getName();
        keyCache.add(kernelName);
        uint64_t id = hash(keyCache.c_str()) % SIZE_CACHE_KERNEL;
        // TODO: Thread safe?

        while (_kernelCache[id].getKernel() && strcmp(_kernelCache[id].getName(), kernelName) != 0)
        {
            id = (id + 1) % SIZE_CACHE_KERNEL;
        }

        if (_kernelCache[id].getKernel())
        {
            kernelPtr = _kernelCache[id].getKernel();
        }
        else
        {
            auto kernelRef = OpenClKernelRef(_clProgramRef->get(), kernelName, status);
            if (status != nullptr && !status->ok())
            {
                return KernelPtr();
            }
            kernelPtr = KernelPtr(new OpenClKernel(_executionTarget, *_clProgramRef, kernelRef));
            _kernelCache[id].setKernel(kernelPtr, kernelName);
        }
        return kernelPtr;
    }

    ~OpenClKernelFactory() DAAL_C11_OVERRIDE {}

protected:
    uint64_t hash(const char * key)
    {
        uint64_t hash    = 5381;
        const char * str = key;
        char c;

        while (c = *str++)
        {
            hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        }
        return hash;
    }

private:
    static const size_t SIZE_CACHE_PROGRAM = 512u;
    static const size_t SIZE_CACHE_KERNEL  = 2048u;
    ProgramCacheEntry _clProgramCache[SIZE_CACHE_PROGRAM];
    KernelCacheEntry _kernelCache[SIZE_CACHE_KERNEL];

    OpenClProgramRef * _clProgramRef;

    ExecutionTargetId _executionTarget;
    cl::sycl::queue & _deviceQueue;
};

class SyclEventImpl : public Base, public SyclEventIface
{
public:
    SyclEventImpl() : _event() {}

    SyclEventImpl(const cl::sycl::event & event) : _event(event) {}

    void wait() { _event.wait(); }

    void waitAndThrow() { _event.wait_and_throw(); }

private:
    cl::sycl::event _event;
};

class SyclExecutionContextImpl : public Base, public ExecutionContextIface
{
public:
    explicit SyclExecutionContextImpl(const cl::sycl::queue & deviceQueue)
        : _deviceQueue(deviceQueue), _kernelFactory(_deviceQueue), _kernelScheduler(_deviceQueue), _event()
    {
        const auto & device                    = _deviceQueue.get_device();
        const cl::sycl::id<3> maxWorkItemSizes = device.get_info<cl::sycl::info::device::max_work_item_sizes>();
        _infoDevice.isCpu                      = device.is_cpu() || device.is_host();
        _infoDevice.max_work_item_sizes_1d     = maxWorkItemSizes[0];
        _infoDevice.max_work_item_sizes_2d     = maxWorkItemSizes[1];
        _infoDevice.max_work_group_size        = device.get_info<cl::sycl::info::device::max_work_group_size>();
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

    void axpy(const uint32_t n, const double a, const UniversalBuffer x_buffer, const int incx, UniversalBuffer y_buffer, const int incy,
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

    SyclEventIface & copy(UniversalBuffer dest, size_t desOffset, UniversalBuffer src, size_t srcOffset, size_t count,
                          services::Status * status = nullptr, bool isSync = true) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(dest.type() == src.type());
        // TODO: Thread safe?
        try
        {
            _event = BufferCopier::copy(_deviceQueue, dest, desOffset, src, srcOffset, count, isSync);
            return _event;
        }
        catch (cl::sycl::exception const & e)
        {
            convertSyclExceptionToStatus(e, status);
            return _event;
        }
    }

    SyclEventIface & fill(UniversalBuffer dest, double value, services::Status * status = nullptr, bool isSync = true) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        try
        {
            _event = BufferFiller::fill(_deviceQueue, dest, value, isSync);
            return _event;
        }
        catch (cl::sycl::exception const & e)
        {
            convertSyclExceptionToStatus(e, status);
            return _event;
        }
    }

    ClKernelFactoryIface & getClKernelFactory() DAAL_C11_OVERRIDE { return _kernelFactory; }

    InfoDevice & getInfoDevice() DAAL_C11_OVERRIDE { return _infoDevice; }

    SyclEventIface & copy(UniversalBuffer dest, size_t desOffset, void * src, size_t srcOffset, size_t count, services::Status * status = nullptr,
                          bool isSync = true) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        try
        {
            _event = ArrayCopier::copy(_deviceQueue, dest, desOffset, src, srcOffset, count, isSync);
            return _event;
        }
        catch (cl::sycl::exception const & e)
        {
            convertSyclExceptionToStatus(e, status);
            return _event;
        }
    }

private:
    cl::sycl::queue _deviceQueue;
    OpenClKernelFactory _kernelFactory;
    SyclKernelScheduler _kernelScheduler;
    InfoDevice _infoDevice;
    SyclEventImpl _event;
};

/** } */
} // namespace interface1

using interface1::SyclEventImpl;
using interface1::SyclExecutionContextImpl;

} // namespace internal
} // namespace oneapi
} // namespace daal

    #endif
#endif // DAAL_SYCL_INTERFACE
