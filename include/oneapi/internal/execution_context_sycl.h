/* file: execution_context_sycl.h */
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

#ifdef DAAL_SYCL_INTERFACE
#ifndef __DAAL_ONEAPI_INTERNAL_EXECUTION_CONTEXT_SYCL_H__
#define __DAAL_ONEAPI_INTERNAL_EXECUTION_CONTEXT_SYCL_H__

#include <vector>
#include <cstring>
#include <CL/cl.h>
#include <CL/sycl.hpp>

#if defined(__SYCL_COMPILER_VERSION) && (__SYCL_COMPILER_VERSION >= 20191024)
  #define DAAL_SYCL_INTERFACE_REVERSED_RANGE
#endif

#include "services/daal_string.h"
#include "oneapi/internal/execution_context.h"
#include "oneapi/internal/types_utils_cxx11.h"
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
/** @ingroup oneapi_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__OPENCLRESOURCEREF"></a>
 *  \brief RAII container for OpenCL* resource
 */
template <typename OpenClType,
          typename OpenClRetain,
          typename OpenClRelease>
class OpenClResourceRef : public Base
{
public:
    OpenClResourceRef() : _resource(nullptr) {}

    OpenClResourceRef(OpenClType &resource) : _resource(resource) {}

    OpenClResourceRef(const OpenClResourceRef &other)
    {
        _resource = other._resource;
        OpenClRetain()(_resource);
    }

    OpenClResourceRef(OpenClResourceRef &&other)
    {
        _resource = other.release();
    }

    ~OpenClResourceRef()
    {
        reset();
    }

    operator bool() const
    {
        return get() == nullptr;
    }

    OpenClResourceRef &operator=(OpenClResourceRef other)
    {
        return swap(other);
    }

    OpenClResourceRef &operator=(OpenClResourceRef &&other)
    {
        _resource = other.release();
        return *this;
    }

    OpenClResourceRef &swap(OpenClResourceRef &other)
    {
        OpenClType tmp = _resource;
        _resource = other._resource;
        other._resource = tmp;
        return *this;
    }

    OpenClType release()
    {
        OpenClType tmp = _resource;
        _resource = nullptr;
        return tmp;
    }

    void reset(OpenClType resource = nullptr)
    {
        OpenClRelease()(_resource);
        _resource = resource;
    }

    OpenClType get() const
    {
        return _resource;
    }

private:
    OpenClType _resource;
};

#define DAAL_DECLARE_OPENCL_OPERATOR(type_, name_) \
    struct OpenCl##name_                           \
    {                                              \
        void operator()(type_ p) { cl##name_(p); } \
    }

DAAL_DECLARE_OPENCL_OPERATOR(cl_program, RetainProgram);
DAAL_DECLARE_OPENCL_OPERATOR(cl_program, ReleaseProgram);
DAAL_DECLARE_OPENCL_OPERATOR(cl_kernel, RetainKernel);
DAAL_DECLARE_OPENCL_OPERATOR(cl_kernel, ReleaseKernel);

#undef DAAL_DECLARE_OPENCL_OPERATOR

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__OPENCLPROGRAMREF"></a>
 *  \brief RAII container for OpenCL* program
 */
class OpenClProgramRef : public OpenClResourceRef<
                             cl_program, OpenClRetainProgram, OpenClReleaseProgram>
{
public:
    OpenClProgramRef() = default;

    explicit OpenClProgramRef(cl_context clContext,
                              cl_device_id clDevice,
                              const char *programName,
                              const char *programSrc,
                              const char *options,
                              services::Status *status = nullptr)
    {
        _progamName = programName;
        cl_int err = 0;
        const char *sources[] = {programSrc};
        const size_t lengths[] = {std::strlen(programSrc)};
        reset(clCreateProgramWithSource(clContext, 1, sources, lengths, &err));
        DAAL_CHECK_OPENCL(err, status)

        err = clBuildProgram(get(), 1, &clDevice, options, nullptr, nullptr);

        #ifdef DAAL_EXECUTION_CONTEXT_VERBOSE
            if (err == CL_BUILD_PROGRAM_FAILURE)
            {
                size_t logLen = 0;
                clGetProgramBuildInfo(get(), clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
                services::Collection<char> buildLogCollection(logLen);
                char *buildLog = buildLogCollection.data();
                if (buildLog == nullptr)
                {
                    printf("common_ocl.ocdBuildProgramFromFile() - Heap Overflow! Cannot allocate space for buildLog.");
                    DAAL_CHECK_OPENCL(err, status)
                }
                clGetProgramBuildInfo(get(), clDevice, CL_PROGRAM_BUILD_LOG,
                    logLen, (void *)buildLog, nullptr);
                printf("CL Error %d: Failed to build program! Log:\n%s", err, buildLog);
            }
        #endif
        DAAL_CHECK_OPENCL(err, status)
    }

    const char *getName() const
    {
        return _progamName.c_str();
    }

private:
    services::String _progamName;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__OPENCLKERNELREF"></a>
 *  \brief RAII container for OpenCL* kernel
 */
class OpenClKernelRef : public OpenClResourceRef<
                            cl_kernel, OpenClRetainKernel, OpenClReleaseKernel>
{
public:
    OpenClKernelRef() = default;

    explicit OpenClKernelRef(cl_program clProgram,
                             const char *kernelName,
                             services::Status *status = nullptr)
    {
        cl_int err = 0;
        reset(clCreateKernel(clProgram, kernelName, &err));
        DAAL_CHECK_OPENCL(err, status)
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__OPENCLKERNEL"></a>
 *  \brief Adapter for OpenCL* kernel
 */
class OpenClKernel : public Base, public KernelIface
{
public:
    explicit OpenClKernel(ExecutionTargetId executionTarget,
                          const OpenClProgramRef &programRef,
                          const OpenClKernelRef &kernelRef) :
        _executionTarget(executionTarget),
        _clProgramRef(programRef),
        _clKernelRef(kernelRef) {}

    void schedule(KernelSchedulerIface &scheduler,
                  const KernelRange &range,
                  const KernelArguments &args,
                  services::Status *status = nullptr) const DAAL_C11_OVERRIDE
    {
        scheduler.schedule(*this, range, args, status);
    }

    void schedule(KernelSchedulerIface &scheduler,
                  const KernelNDRange &range,
                  const KernelArguments &args,
                  services::Status *status = nullptr) const DAAL_C11_OVERRIDE
    {
        scheduler.schedule(*this, range, args, status);
    }

    ExecutionTargetId getTarget() const
    {
        return _executionTarget;
    }

    const OpenClKernelRef &getRef() const
    {
        return _clKernelRef;
    }

private:
    ExecutionTargetId _executionTarget;
    OpenClProgramRef _clProgramRef;
    OpenClKernelRef _clKernelRef;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__OPENCLKERNELFACTORY"></a>
 *  \brief Implementation of OpenCL* kernel factory
 */
class OpenClKernelFactory : public Base, public ClKernelFactoryIface
{
public:
    explicit OpenClKernelFactory(cl::sycl::queue &deviceQueue) : _executionTarget(ExecutionTargetIds::unspecified),
                                                                 _deviceQueue(deviceQueue)
    {
        for (size_t i = 0; i < SIZE_CACHE_PROGRAM; i++)
        {
            _clProgramCache[i] = nullptr;
        }
    }

    void build(ExecutionTargetId target,
               const char *key,
               const char *program,
               const char *options = "",
               services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        // TODO Rework of "cache"

        const uint64_t id = hash(key) % SIZE_CACHE_PROGRAM;

        if (_clProgramCache[id])
        {
            _clProgramRef = _clProgramCache[id];
            _executionTarget = target;
        }
        else
        {
            _clProgramCache[id] = new OpenClProgramRef(_deviceQueue.get_context().get(),
                                                        _deviceQueue.get_device().get(),
                                                        key, program,
                                                        options,
                                                        status);
            if (status != nullptr && !status->ok())
            {
                return;
            }
            _clProgramRef = _clProgramCache[id];
            _executionTarget = target;
        }
    }

    KernelPtr getKernel(const char *kernelName,
                        services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(_clProgramRef);
        DAAL_ASSERT(*_clProgramRef);
        KernelPtr kernelPtr;

        services::String keyCache = _clProgramRef->getName();
        keyCache.add(kernelName);
        const uint64_t id = hash(keyCache.c_str()) % SIZE_CACHE_KERNEL;
        // TODO: Thread safe?

        if (_kernelCache[id])
        {
            kernelPtr = _kernelCache[id];
        }
        else
        {
            auto kernelRef = OpenClKernelRef(_clProgramRef->get(), kernelName, status);
            if (status != nullptr && !status->ok())
            {
                return KernelPtr();
            }
            kernelPtr = KernelPtr(new OpenClKernel(_executionTarget, *_clProgramRef, kernelRef));
            _kernelCache[id] = kernelPtr;
        }
        return kernelPtr;
    }

    ~OpenClKernelFactory() DAAL_C11_OVERRIDE
    {
        for (size_t i = 0; i < SIZE_CACHE_PROGRAM; i++)
        {
            if (_clProgramCache[i])
                delete _clProgramCache[i];
        }
    }

protected:
    uint64_t hash(const char *key)
    {
        uint64_t hash = 5381;
        const char *str = key;
        char c;

        while (c = *str++)
        {
            hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        }
        return hash;
    }

private:
    static const size_t SIZE_CACHE_PROGRAM = 512u;
    static const size_t SIZE_CACHE_KERNEL = 2048u;
    OpenClProgramRef *_clProgramCache[SIZE_CACHE_PROGRAM];
    KernelPtr _kernelCache[SIZE_CACHE_KERNEL];

    OpenClProgramRef *_clProgramRef;

    ExecutionTargetId _executionTarget;
    cl::sycl::queue &_deviceQueue;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__SYCLBUFFERSTORAGE"></a>
 *  \brief Storage for sycl buffer
 */
class SyclBufferStorage
{
public:
    template <typename T>
    void add(const cl::sycl::buffer<T, 1> &buffer)
    {
        _buffers.push_back(services::internal::Any(buffer));
    }

private:
    std::vector<services::internal::Any> _buffers;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__SYCLKERNELSCHEDULER"></a>
 *  \brief Scheduler for SYCL* kernels
 */
class SyclKernelScheduler : public Base, public KernelSchedulerIface
{
private:
    class ArgHandler
    {
    public:
        explicit ArgHandler(cl::sycl::handler &cgh,
                            const KernelArgument &arg,
                            size_t idx,
                            SyclBufferStorage &storage,
                            services::Status *status = nullptr) : _cgh(cgh), _arg(arg), _idx(idx), _storage(storage), _status(status) {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            try
            {
                // TODO: Implement verbose mode?
                // if (TypeIds::id<T>() == TypeIds::float32)
                // {
                //     if (_arg.isBuffer())
                //     {
                //         printf("ArgHandler: float buffer\n");
                //     }
                //     else
                //     {
                //         printf("ArgHandler: float scalar\n");
                //     }
                // }

                if (_arg.argType() == KernelArgumentTypes::publicBuffer)
                {
                    auto syclBuffer = _arg.get<services::Buffer<T>>().toSycl();
                    _storage.add(syclBuffer); // we need this storage to keep all sycl buffers alive until kernel run
                    switch (_arg.accessMode())
                    {
                    case AccessModeIds::read:
                        _cgh.set_arg((int)_idx, syclBuffer.template get_access<cl::sycl::access::mode::read>(_cgh));
                        break;

                    case AccessModeIds::write:
                        _cgh.set_arg((int)_idx, syclBuffer.template get_access<cl::sycl::access::mode::write>(_cgh));
                        break;

                    case AccessModeIds::readwrite:
                        _cgh.set_arg((int)_idx, syclBuffer.template get_access<cl::sycl::access::mode::read_write>(_cgh));
                        break;
                    }
                }
                else if (_arg.argType() == KernelArgumentTypes::privateBuffer)
                {
                    // Expression _cgh.set_arg(_idx,
                    //                         cl::sycl::accessor<T, 1,
                    //                         cl::sycl::access::mode::read_write,
                    //                         cl::sycl::access::target::local>(bufferSize, _cgh))
                    // does not compile due to bug in Intel SYCL implementation

                    // auto localBuffer = _arg.get<LocalBuffer>();
                    // auto bufferSize = cl::sycl::range<1>(localBuffer.size()*sizeof(T));
                    // _cgh.set_arg(_idx,
                    //              cl::sycl::accessor<T, 1,
                    //                                 cl::sycl::access::mode::read_write,
                    //                                 cl::sycl::access::target::local>(bufferSize, _cgh));

                    DAAL_ASSERT(!"Local buffers are not supported!");
                }
                else if (_arg.argType() == KernelArgumentTypes::publicConstant)
                {
                    // Expression _cgh.set_arg(_idx, _arg.get<T>()) does not compile due to
                    // bug in Intel SYCL implementation
                    T value = _arg.get<T>();
                    _cgh.set_arg((int)_idx, value);
                }
                else
                {
                    if (_status != nullptr)
                    {
                        _status->add(services::ErrorID::ErrorMethodNotImplemented);
                    }
                }
            }
            catch (cl::sycl::exception const &e)
            {
                convertSyclExceptionToStatus(e, _status);
            }
        }

    private:
        cl::sycl::handler &_cgh;
        const KernelArgument &_arg;
        size_t _idx;
        SyclBufferStorage &_storage;
        services::Status *_status;
    };

public:
    explicit SyclKernelScheduler(cl::sycl::queue &deviceQueue) : _deviceQueue(deviceQueue) {}

    void schedule(const OpenClKernel &kernel,
                  const KernelRange &range,
                  const KernelArguments &args,
                  services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        schedule_impl(kernel, range, args, status);
    }

    void schedule(const OpenClKernel &kernel,
                  const KernelNDRange &range,
                  const KernelArguments &args,
                  services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        schedule_impl(kernel, range, args, status);
    }

private:
    template <typename TRange>
    void schedule_impl(const OpenClKernel &kernel,
                       const TRange &range,
                       const KernelArguments &args,
                       services::Status *status = nullptr)
    {
        switch (kernel.getTarget())
        {
        case ExecutionTargetIds::device:
            scheduleOnDevice(_deviceQueue, kernel, range, args, status);
            break;

        case ExecutionTargetIds::host:
            if (status != nullptr)
            {
                status->add(services::ErrorID::ErrorMethodNotImplemented);
            }
            DAAL_ASSERT(!"Not implemented");
            break;

        default:
            if (status != nullptr)
            {
                status->add(services::ErrorID::UnknownError);
            }
            DAAL_ASSERT(!"Unexpected");
            break;
        }
    }

    void scheduleOnDevice(cl::sycl::queue &queue,
                          const OpenClKernel &kernel,
                          const KernelRange &range,
                          const KernelArguments &args,
                          services::Status *status = nullptr)
    {
        try
        {
            if (range.dimensions() == 1)
            {
                schedule(_deviceQueue, kernel, cl::sycl::range<1>(range.upper1()), args, status);
            }
            else if (range.dimensions() == 2)
            {
                #ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
                    schedule(_deviceQueue, kernel, cl::sycl::range<2>(range.upper2(), range.upper1()), args, status);
                #else
                    schedule(_deviceQueue, kernel, cl::sycl::range<2>(range.upper1(), range.upper2()), args, status);
                #endif
            }
            else if (range.dimensions() == 3)
            {
                schedule(_deviceQueue, kernel, cl::sycl::range<3>(range.upper1(), range.upper2(), range.upper3()), args, status);
            }
            else
            {
                status->add(services::ErrorMethodNotImplemented);
            }
        }
        catch (cl::sycl::exception const &e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }

    void scheduleOnDevice(cl::sycl::queue &queue,
                          const OpenClKernel &kernel,
                          const KernelNDRange &range,
                          const KernelArguments &args,
                          services::Status *status = nullptr)
    {
        try
        {
            if (range.dimensions() == 1)
            {
                cl::sycl::range<1> global(range.global().upper1());
                cl::sycl::range<1> local(range.local().upper1());
                cl::sycl::nd_range<1> r(global, local);

                schedule(_deviceQueue, kernel, r, args, status);
            }
            else if (range.dimensions() == 2)
            {
                #ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
                    cl::sycl::range<2> global(range.global().upper2(), range.global().upper1());
                    cl::sycl::range<2> local(range.local().upper2(), range.local().upper1());
                #else
                    cl::sycl::range<2> global(range.global().upper1(), range.global().upper2());
                    cl::sycl::range<2> local(range.local().upper1(), range.local().upper2());
                #endif
                cl::sycl::nd_range<2> r(global, local);

                schedule(_deviceQueue, kernel, r, args, status);
            }

            /** This case was never used. Check order of indexing for 3D NDRange */
            else if (range.dimensions() == 3)
            {
                cl::sycl::range<3> global(range.global().upper1(), range.global().upper2(), range.global().upper3());
                cl::sycl::range<3> local(range.local().upper1(), range.local().upper2(), range.local().upper3());
                cl::sycl::nd_range<3> r(global, local);

                schedule(_deviceQueue, kernel, r, args, status);
            }
            else
            {
                status->add(services::ErrorMethodNotImplemented);
            }
        }
        catch (cl::sycl::exception const &e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }

    template <typename TRange>
    void schedule(cl::sycl::queue &queue,
                  const OpenClKernel &kernel,
                  const TRange &range,
                  const KernelArguments &args,
                  services::Status *status = nullptr)
    {
        try
        {
            // TODO: Implement verbose mode?
            cl::sycl::kernel syclKernel(kernel.getRef().get(), queue.get_context());
            SyclBufferStorage syclStorage;
            cl::sycl::event event = queue.submit([&](cl::sycl::handler &cgh) {
                passArguments(cgh, args, syclStorage);
                cgh.parallel_for(range, syclKernel);
            });
            event.wait_and_throw();
        }
        catch (cl::sycl::exception const &e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }

    void passArguments(cl::sycl::handler &cgh, const KernelArguments &args, SyclBufferStorage &storage)
    {
        for (size_t i = 0; i < args.size(); i++)
        {
            const auto &arg = args.get(i);
            TypeDispatcher::dispatch(arg.dataType(), ArgHandler(cgh, arg, i, storage));
        }
    }

private:
    cl::sycl::queue &_deviceQueue;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__SYCLEXECUTIONCONTEXTIMPL"></a>
 *  \brief Implementation of SYCL* execution context
 */
class SyclExecutionContextImpl : public Base, public ExecutionContextIface
{
public:
    explicit SyclExecutionContextImpl(const cl::sycl::queue &deviceQueue) : _deviceQueue(deviceQueue),
                                                                            _kernelFactory(_deviceQueue),
                                                                            _kernelScheduler(_deviceQueue)
    {
        const auto &device = _deviceQueue.get_device();
        const cl::sycl::id<3> maxWorkItemSizes =
            device.get_info<cl::sycl::info::device::max_work_item_sizes>();
        _infoDevice.isCpu = device.is_cpu() || device.is_host();
        _infoDevice.max_work_item_sizes_1d = maxWorkItemSizes[0];
        _infoDevice.max_work_item_sizes_2d = maxWorkItemSizes[1];
        _infoDevice.max_work_group_size =
            device.get_info<cl::sycl::info::device::max_work_group_size>();
    }

    void run(const KernelRange &range,
             const KernelPtr &kernel,
             const KernelArguments &args,
             services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        // TODO: Check for input arguments
        // TODO: Need to save reference to kernel to prevent
        //       releasing in case of asynchronous execution?
        kernel->schedule(_kernelScheduler, range, args, status);
    }

    void run(const KernelNDRange &range,
             const KernelPtr &kernel,
             const KernelArguments &args,
             services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        // TODO: Check for input arguments
        // TODO: Need to save reference to kernel to prevent
        //       releasing in case of asynchronous execution?
        kernel->schedule(_kernelScheduler, range, args, status);
    }

    void gemm(math::Transpose transa, math::Transpose transb,
              size_t m, size_t n, size_t k,
              double alpha,
              const UniversalBuffer &a_buffer, size_t lda, size_t offsetA,
              const UniversalBuffer &b_buffer, size_t ldb, size_t offsetB,
              double beta,
              UniversalBuffer &c_buffer, size_t ldc, size_t offsetC,
              services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(a_buffer.type() == b_buffer.type());
        DAAL_ASSERT(b_buffer.type() == c_buffer.type());

        // TODO: Check for input arguments
        math::GemmExecutor::run(_deviceQueue, transa, transb, m, n, k, alpha, a_buffer, lda, offsetA,
                                b_buffer, ldb, offsetB, beta, c_buffer, ldc, offsetC, status);
    }

    void syrk(math::UpLo upper_lower,
              math::Transpose trans,
              size_t n, size_t k,
              double alpha,
              const UniversalBuffer &a_buffer, size_t lda, size_t offsetA,
              double beta,
              UniversalBuffer &c_buffer, size_t ldc, size_t offsetC,
              services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(a_buffer.type() == c_buffer.type());

        math::SyrkExecutor::run(_deviceQueue, upper_lower, trans, n, k, alpha, a_buffer, lda, offsetA,
                                beta, c_buffer, ldc, offsetC, status);
    }

    void potrf(math::UpLo uplo, size_t n,
               UniversalBuffer &a_buffer, size_t lda, services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        math::PotrfExecutor::run(_deviceQueue, uplo, n, a_buffer, lda, status);
    }

    void potrs(math::UpLo uplo, size_t n, size_t ny,
               UniversalBuffer &a_buffer, size_t lda,
               UniversalBuffer &b_buffer, size_t ldb, services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(a_buffer.type() == b_buffer.type());
        math::PotrsExecutor::run(_deviceQueue, uplo, n, ny, a_buffer, lda, b_buffer, ldb, status);
    }

    UniversalBuffer allocate(TypeId type,
                             size_t bufferSize,
                             services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        try
        {
            auto buffer = BufferAllocator::allocate(type, bufferSize);
            return buffer;
        }
        catch (cl::sycl::exception const &e)
        {
            convertSyclExceptionToStatus(e, status);
            return UniversalBuffer();
        }
    }

    void copy(UniversalBuffer dest,
              size_t desOffset,
              UniversalBuffer src,
              size_t srcOffset,
              size_t count,
              services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(dest.type() == src.type());
        // TODO: Thread safe?
        try
        {
            BufferCopier::copy(_deviceQueue, dest,
                               desOffset, src, srcOffset, count);
        }
        catch (cl::sycl::exception const &e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }

    void fill(UniversalBuffer dest,
              double value,
              services::Status *status = nullptr) DAAL_C11_OVERRIDE
    {
        // TODO: Thread safe?
        try
        {
            BufferFiller::fill(_deviceQueue, dest, value);
        }
        catch (cl::sycl::exception const &e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }

    ClKernelFactoryIface &getClKernelFactory() DAAL_C11_OVERRIDE
    {
        return _kernelFactory;
    }

    InfoDevice &getInfoDevice() DAAL_C11_OVERRIDE
    {
        return _infoDevice;
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
