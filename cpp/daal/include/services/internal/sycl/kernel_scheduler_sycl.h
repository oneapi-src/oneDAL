/* file: kernel_scheduler_sycl.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_KERNEL_SCHEDULER_SYCL_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_KERNEL_SCHEDULER_SYCL_H__

#ifndef DAAL_SYCL_INTERFACE
    #error "DAAL_SYCL_INTERFACE must be defined to include this file"
#endif

#include <cstring>

#include <CL/cl.h>
#include <sycl/sycl.hpp>

#include "services/internal/sycl/error_handling_sycl.h"
#include "services/internal/sycl/execution_context.h"
#include "services/internal/sycl/buffer_utils_sycl.h"

#ifndef DAAL_DISABLE_LEVEL_ZERO
    #include "services/internal/sycl/level_zero_module_sycl.h"
#endif

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
template <typename OpenClType, typename OpenClRetain, typename OpenClRelease>
class OpenClResourceRef : public Base
{
public:
    OpenClResourceRef() : _resource(nullptr) {}

    OpenClResourceRef(OpenClType & resource) : _resource(resource) {}

    OpenClResourceRef(const OpenClResourceRef & other)
    {
        _resource = other._resource;
        OpenClRetain()(_resource);
    }

    OpenClResourceRef(OpenClResourceRef && other) { _resource = other.release(); }

    ~OpenClResourceRef() { reset(); }

    operator bool() const { return get() == nullptr; }

    OpenClResourceRef & operator=(OpenClResourceRef other) { return swap(other); }

    OpenClResourceRef & operator=(OpenClResourceRef && other)
    {
        _resource = other.release();
        return *this;
    }

    OpenClResourceRef & swap(OpenClResourceRef & other)
    {
        OpenClType tmp  = _resource;
        _resource       = other._resource;
        other._resource = tmp;
        return *this;
    }

    OpenClType release()
    {
        OpenClType tmp = _resource;
        _resource      = nullptr;
        return tmp;
    }

    void reset(OpenClType resource = nullptr)
    {
        OpenClRelease()(_resource);
        _resource = resource;
    }

    OpenClType get() const { return _resource; }

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
DAAL_DECLARE_OPENCL_OPERATOR(cl_context, RetainContext);
DAAL_DECLARE_OPENCL_OPERATOR(cl_context, ReleaseContext);
DAAL_DECLARE_OPENCL_OPERATOR(cl_device_id, RetainDevice);
DAAL_DECLARE_OPENCL_OPERATOR(cl_device_id, ReleaseDevice);

#undef DAAL_DECLARE_OPENCL_OPERATOR

typedef OpenClResourceRef<cl_device_id, OpenClRetainDevice, OpenClReleaseDevice> OpenClDeviceRef;

class OpenClContextRef : public OpenClResourceRef<cl_context, OpenClRetainContext, OpenClReleaseContext>
{
public:
    OpenClContextRef() = default;

    explicit OpenClContextRef(cl_device_id clDevice, Status & status) : _clDeviceRef(clDevice)
    {
        cl_int err = 0;
        reset(clCreateContext(nullptr, 1, &clDevice, nullptr, nullptr, &err));
        DAAL_CHECK_OPENCL(err, status)
    }

    OpenClDeviceRef getDeviceRef() { return _clDeviceRef; }

private:
    OpenClDeviceRef _clDeviceRef;
};

#ifndef DAAL_DISABLE_LEVEL_ZERO
class LevelZeroOpenClInteropContext : public Base
{
public:
    LevelZeroOpenClInteropContext() = default;

    LevelZeroOpenClInteropContext(const LevelZeroOpenClInteropContext &) = delete;

    explicit LevelZeroOpenClInteropContext(::sycl::queue & deviceQueue, Status & status) { reset(deviceQueue, status); }

    void reset(::sycl::queue & deviceQueue, Status & status)
    {
        cl_device_id clDevice;
        findDevice(&clDevice, deviceQueue.get_device().get_info< ::sycl::info::device::vendor_id>(),
                   deviceQueue.get_device().get_info< ::sycl::info::device::max_clock_frequency>(), status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        _clDeviceRef.reset(clDevice);

        cl_int err = 0;
        _clContextRef.reset(clCreateContext(nullptr, 1, &clDevice, nullptr, nullptr, &err));
        DAAL_CHECK_OPENCL(err, status)
    }

    void findDevice(cl_device_id * pClDevice, unsigned int vendor_id, unsigned int frq, Status & status)
    {
        constexpr cl_uint maxPlatforms = 16;
        cl_platform_id platIds[maxPlatforms];
        cl_uint nplat, ndev;

        DAAL_CHECK_OPENCL(clGetPlatformIDs(maxPlatforms, platIds, &nplat), status);

        for (cl_uint pidx = 0; pidx < nplat && pidx < maxPlatforms; pidx++)
        {
            if (clGetDeviceIDs(platIds[pidx], CL_DEVICE_TYPE_GPU, 1, pClDevice, &ndev) == CL_SUCCESS)
            {
                if (ndev > 0)
                {
                    cl_uint dVid = 0;
                    clGetDeviceInfo(*pClDevice, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &dVid, nullptr);

                    cl_uint dFrq = 0;
                    clGetDeviceInfo(*pClDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &dFrq, nullptr);

                    if (dVid == vendor_id && dFrq == frq) return;
                }
            }
        }

        status |= ErrorDeviceSupportNotImplemented;
    }

    OpenClDeviceRef & getOpenClDeviceRef() { return _clDeviceRef; }
    OpenClContextRef & getOpenClContextRef() { return _clContextRef; }

private:
    OpenClContextRef _clContextRef;
    OpenClDeviceRef _clDeviceRef;
};
#endif // DAAL_DISABLE_LEVEL_ZERO

class OpenClProgramRef : public OpenClResourceRef<cl_program, OpenClRetainProgram, OpenClReleaseProgram>
{
public:
    static SharedPtr<OpenClProgramRef> create(cl_context clContext, cl_device_id clDevice, const char * programName, const char * programSrc,
                                              const char * options, Status & status)
    {
        auto ptr = new OpenClProgramRef();
        if (!ptr)
        {
            status |= ErrorMemoryAllocationFailed;
            return SharedPtr<OpenClProgramRef>();
        }
        ptr->initOpenClProgramRef(clContext, clDevice, programName, programSrc, options, status);
        return SharedPtr<OpenClProgramRef>(ptr);
    }

#ifndef DAAL_DISABLE_LEVEL_ZERO
    static SharedPtr<OpenClProgramRef> create(cl_context clContext, cl_device_id clDevice, ::sycl::queue & deviceQueue, const char * programName,
                                              const char * programSrc, const char * options, Status & status)
    {
        auto ptr = new OpenClProgramRef();
        if (!ptr)
        {
            status |= ErrorMemoryAllocationFailed;
            return SharedPtr<OpenClProgramRef>();
        }
        ptr->initOpenClProgramRef(clContext, clDevice, programName, programSrc, options, status);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, SharedPtr<OpenClProgramRef>());

        ptr->initModuleLevelZero(deviceQueue, status);
        return SharedPtr<OpenClProgramRef>(ptr);
    }
#endif // DAAL_DISABLE_LEVEL_ZERO

#ifndef DAAL_DISABLE_LEVEL_ZERO
    ZeModulePtr getModuleLevelZeroPtr() const
    {
        return _moduleLevelZeroPtr;
    }
#endif // DAAL_DISABLE_LEVEL_ZERO

    const String & getName() const
    {
        return _programName;
    }

private:
    OpenClProgramRef() = default;

    void initOpenClProgramRef(cl_context clContext, cl_device_id clDevice, const char * programName, const char * programSrc, const char * options,
                              Status & status)
    {
        _programName = programName;
        DAAL_CHECK_COND_ERROR(_programName.c_str(), status, ErrorMemoryAllocationFailed);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        cl_int err             = 0;
        const char * sources[] = { programSrc };
        const size_t lengths[] = { std::strlen(programSrc) };
        reset(clCreateProgramWithSource(clContext, 1, sources, lengths, &err));
        DAAL_CHECK_OPENCL(err, status)

        err = clBuildProgram(get(), 1, &clDevice, options, nullptr, nullptr);

        DAAL_ASSERT_DECL(if (err == CL_BUILD_PROGRAM_FAILURE) {
            size_t logLen = 0;
            clGetProgramBuildInfo(get(), clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
            String buildLog(logLen);
            if (buildLog.c_str())
            {
                clGetProgramBuildInfo(get(), clDevice, CL_PROGRAM_BUILD_LOG, logLen, (void *)(buildLog.c_str()), nullptr);
                fprintf(stderr, "Failed to build OpenCL program (%d):\n%s", err, buildLog.c_str());
            }
        })

        DAAL_CHECK_OPENCL(err, status)
    }

#ifndef DAAL_DISABLE_LEVEL_ZERO
    void initModuleLevelZero(::sycl::queue & deviceQueue, Status & status)
    {
        size_t binarySize = 0;
        DAAL_CHECK_OPENCL(clGetProgramInfo(get(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, nullptr), status);

        Collection<byte> binaryCollection(binarySize);
        DAAL_CHECK_COND_ERROR(binaryCollection.data(), status, ErrorMemoryAllocationFailed);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        byte * binary = binaryCollection.data();
        DAAL_CHECK_OPENCL(clGetProgramInfo(get(), CL_PROGRAM_BINARIES, sizeof(binary), &binary, nullptr), status);

        _moduleLevelZeroPtr = ZeModule::create(deviceQueue, binarySize, binary, status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
    }
#endif // DAAL_DISABLE_LEVEL_ZERO

private:
    String _programName;

#ifndef DAAL_DISABLE_LEVEL_ZERO
    ZeModulePtr _moduleLevelZeroPtr;
#endif
};

class OpenClKernelRef : public OpenClResourceRef<cl_kernel, OpenClRetainKernel, OpenClReleaseKernel>
{
public:
    OpenClKernelRef() = default;

    explicit OpenClKernelRef(cl_program clProgram, const String & kernelName, Status & status)
    {
        cl_int err = 0;
        reset(clCreateKernel(clProgram, kernelName.c_str(), &err));
        DAAL_CHECK_OPENCL(err, status)
    }
};

#ifndef DAAL_DISABLE_LEVEL_ZERO
class OpenClKernelLevelZeroRef : public Base
{
public:
    OpenClKernelLevelZeroRef() = default;

    explicit OpenClKernelLevelZeroRef(const OpenClProgramRef & programRef, const String & kernelName, Status & status)
    {
        _kernelLevelZeroPtr = programRef.getModuleLevelZeroPtr()->createKernel(kernelName.c_str(), status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
    }

    ZeKernelPtr getKernelLevelZeroPtr() const { return _kernelLevelZeroPtr; }

private:
    ZeKernelPtr _kernelLevelZeroPtr;
};
#endif // DAAL_DISABLE_LEVEL_ZERO

class OpenClKernel : public Base, public KernelIface
{
public:
    explicit OpenClKernel(ExecutionTargetId executionTarget, const OpenClProgramRef & programRef)
        : _executionTarget(executionTarget), _clProgramRef(programRef)
    {}

    void schedule(KernelSchedulerIface & scheduler, const KernelRange & range, const KernelArguments & args, Status & status) const DAAL_C11_OVERRIDE
    {
        scheduler.schedule(*this, range, args, status);
    }

    void schedule(KernelSchedulerIface & scheduler, const KernelNDRange & range, const KernelArguments & args,
                  Status & status) const DAAL_C11_OVERRIDE
    {
        scheduler.schedule(*this, range, args, status);
    }

    ExecutionTargetId getTarget() const { return _executionTarget; }

    virtual ::sycl::kernel toSycl(const ::sycl::context & ctx) const = 0;

    const OpenClProgramRef & getProgramRef() const { return _clProgramRef; }

private:
    ExecutionTargetId _executionTarget;
    OpenClProgramRef _clProgramRef;
};

class OpenClKernelNative : public OpenClKernel
{
public:
    static SharedPtr<OpenClKernelNative> create(ExecutionTargetId executionTarget, const OpenClProgramRef & programRef,
                                                const OpenClKernelRef & kernelRef, Status & status)
    {
        auto * ptr = new OpenClKernelNative(executionTarget, programRef, kernelRef);
        if (!ptr) status |= ErrorMemoryAllocationFailed;
        return SharedPtr<OpenClKernelNative>(ptr);
    }

    ::sycl::kernel toSycl(const ::sycl::context & ctx) const DAAL_C11_OVERRIDE
    {
        return ::sycl::make_kernel< ::sycl::backend::opencl>(_clKernelRef.get(), ctx);
    }

private:
    explicit OpenClKernelNative(ExecutionTargetId executionTarget, const OpenClProgramRef & programRef, const OpenClKernelRef & kernelRef)
        : OpenClKernel(executionTarget, programRef), _clKernelRef(kernelRef)
    {}

    OpenClKernelRef _clKernelRef;
};

#ifndef DAAL_DISABLE_LEVEL_ZERO
class OpenClKernelLevelZero : public OpenClKernel
{
public:
    static SharedPtr<OpenClKernelLevelZero> create(ExecutionTargetId executionTarget, const OpenClProgramRef & programRef,
                                                   const OpenClKernelLevelZeroRef & kernelRef, Status & status)
    {
        auto * ptr = new OpenClKernelLevelZero(executionTarget, programRef, kernelRef);
        if (!ptr) status |= ErrorMemoryAllocationFailed;
        return SharedPtr<OpenClKernelLevelZero>(ptr);
    }

    ::sycl::kernel toSycl(const ::sycl::context & ctx) const DAAL_C11_OVERRIDE
    {
        using namespace ::sycl;
        kernel_bundle<bundle_state::executable> _kernelBundle = make_kernel_bundle<backend::ext_oneapi_level_zero, bundle_state::executable>(
            { getProgramRef().getModuleLevelZeroPtr()->get(), ext::oneapi::level_zero::ownership::keep }, ctx);
        return make_kernel<backend::ext_oneapi_level_zero>(
            { _kernelBundle, _zeKernelRef.getKernelLevelZeroPtr()->get(), ext::oneapi::level_zero::ownership::keep }, ctx);
    }

private:
    OpenClKernelLevelZero(ExecutionTargetId executionTarget, const OpenClProgramRef & programRef, const OpenClKernelLevelZeroRef & kernelRef)
        : OpenClKernel(executionTarget, programRef), _zeKernelRef(kernelRef)
    {}

    OpenClKernelLevelZeroRef _zeKernelRef;
};
#endif // DAAL_DISABLE_LEVEL_ZERO

class UsmPointerStorage
{
public:
    UsmPointerStorage()                                      = default;
    UsmPointerStorage(const UsmPointerStorage &)             = delete;
    UsmPointerStorage & operator=(const UsmPointerStorage &) = delete;

    template <typename T>
    bool add(const SharedPtr<T> & usmPointer)
    {
        return _pointers.safe_push_back(Any(usmPointer));
    }

private:
    Collection<Any> _pointers;
};

class SyclKernelSchedulerArgHandler
{
public:
    SyclKernelSchedulerArgHandler(::sycl::queue & queue, ::sycl::handler & handler, UsmPointerStorage & storage, size_t argumentIndex,
                                  const KernelArgument & arg)
        : _queue(queue), _handler(handler), _storage(storage), _argumentIndex(argumentIndex), _argument(arg)
    {}

    template <typename T>
    void operator()(Typelist<T>, Status & status)
    {
        switch (_argument.argType())
        {
        case KernelArgumentTypes::publicBuffer: return handlePublicBuffer<T>(status);
        case KernelArgumentTypes::privateBuffer: return handlePrivateBuffer<T>(status);
        case KernelArgumentTypes::publicConstant: return handlePublicConstant<T>(status);
        default: DAAL_ASSERT(!"Unexpected kernel argument type");
        }
    }

private:
    template <typename T>
    void handlePublicBuffer(Status & status)
    {
        auto service_buffer = _argument.get<Buffer<T> >();
#ifdef DAAL_SYCL_INTERFACE_USM
        switch (_argument.accessMode())
        {
        case AccessModeIds::read: return handlePublicBuffer<data_management::readOnly>(service_buffer, status);

        case AccessModeIds::write: return handlePublicBuffer<data_management::readWrite>(service_buffer, status);

        case AccessModeIds::readwrite: return handlePublicBuffer<data_management::readWrite>(service_buffer, status);

        default: DAAL_ASSERT(!"Unexpected buffer access mode");
        }
#else
        static_assert(false, "USM memory support is required for kernel execution");
#endif
    }

#ifdef DAAL_SYCL_INTERFACE_USM
    template <data_management::ReadWriteMode mode, typename T>
    void handlePublicBuffer(Buffer<T> & buffer, Status & status)
    {
        auto shared_pointer = buffer.toUSM(_queue, mode, status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        // Note: we need this storage to keep all usm shared pointers alive
        // while the kernel is running
        if (!_storage.add(shared_pointer))
        {
            status |= ErrorMemoryAllocationFailed;
            return;
        }

        _handler.set_arg((int)_argumentIndex, shared_pointer.get());
    }
#endif

    template <typename T>
    void handlePrivateBuffer(Status & status)
    {
        DAAL_ASSERT(!"Local buffers are not supported");
    }

    template <typename T>
    void handlePublicConstant(Status & status)
    {
        T value = _argument.get<T>();
        _handler.set_arg((int)_argumentIndex, value);
    }

    ::sycl::queue & _queue;
    ::sycl::handler & _handler;
    UsmPointerStorage & _storage;
    size_t _argumentIndex;
    const KernelArgument & _argument;
};

template <int dim>
inline ::sycl::range<dim> convertToSyclRange(const KernelRange &);

template <>
inline ::sycl::range<1> convertToSyclRange<1>(const KernelRange & r)
{
    return ::sycl::range<1>(r.upper1());
}

template <>
inline ::sycl::range<2> convertToSyclRange<2>(const KernelRange & r)
{
#ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
    return ::sycl::range<2>(r.upper2(), r.upper1());
#else
    return ::sycl::range<2>(r.upper1(), r.upper2());
#endif
}

template <>
inline ::sycl::range<3> convertToSyclRange<3>(const KernelRange & r)
{
#ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
    return ::sycl::range<3>(r.upper3(), r.upper2(), r.upper1());
#else
    return ::sycl::range<3>(r.upper1(), r.upper2(), r.upper3());
#endif
}

template <int dim>
inline ::sycl::nd_range<dim> convertToSyclRange(const KernelNDRange &);

template <>
inline ::sycl::nd_range<1> convertToSyclRange<1>(const KernelNDRange & r)
{
    return ::sycl::nd_range<1>(::sycl::range<1>(r.global().upper1()), ::sycl::range<1>(r.local().upper1()));
}

template <>
inline ::sycl::nd_range<2> convertToSyclRange<2>(const KernelNDRange & r)
{
    return ::sycl::nd_range<2>(
#ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
        ::sycl::range<2>(r.global().upper2(), r.global().upper1()), ::sycl::range<2>(r.local().upper2(), r.local().upper1())
#else
        ::sycl::range<2>(r.global().upper1(), r.global().upper2()), ::sycl::range<2>(r.local().upper1(), r.local().upper2())
#endif
    );
}

template <>
inline ::sycl::nd_range<3> convertToSyclRange<3>(const KernelNDRange & r)
{
    return ::sycl::nd_range<3>(
#ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
        ::sycl::range<3>(r.global().upper3(), r.global().upper2(), r.global().upper1()),
        ::sycl::range<3>(r.local().upper3(), r.local().upper2(), r.local().upper1())
#else
        ::sycl::range<3>(r.global().upper1(), r.global().upper2(), r.global().upper3()),
        ::sycl::range<3>(r.local().upper1(), r.local().upper2(), r.local().upper3())
#endif
    );
}

class SyclKernelScheduler : public Base, public KernelSchedulerIface
{
public:
    explicit SyclKernelScheduler(::sycl::queue & deviceQueue) : _queue(deviceQueue) {}

    void schedule(const OpenClKernel & kernel, const KernelRange & range, const KernelArguments & args, Status & status) DAAL_C11_OVERRIDE
    {
        scheduleImpl(range, kernel, args, status);
    }

    void schedule(const OpenClKernel & kernel, const KernelNDRange & range, const KernelArguments & args, Status & status) DAAL_C11_OVERRIDE
    {
        scheduleImpl(range, kernel, args, status);
    }

private:
    template <typename Range>
    void scheduleImpl(const Range & range, const OpenClKernel & kernel, const KernelArguments & args, Status & status)
    {
        switch (kernel.getTarget())
        {
        case ExecutionTargetIds::device: return scheduleOnDevice(range, kernel, args, status);

        case ExecutionTargetIds::host: status |= ErrorMethodNotImplemented; return;

        default: DAAL_ASSERT(!"Unexpected execution target");
        }
    }

    template <typename Range>
    void scheduleOnDevice(const Range & range, const OpenClKernel & kernel, const KernelArguments & args, Status & status)
    {
        switch (range.dimensions())
        {
        case 1: return scheduleSycl(convertToSyclRange<1>(range), kernel, args, status);
        case 2: return scheduleSycl(convertToSyclRange<2>(range), kernel, args, status);
        case 3: return scheduleSycl(convertToSyclRange<3>(range), kernel, args, status);
        default: DAAL_ASSERT(!"Unexpected number of dimensions");
        }
    }

    template <typename Range>
    void scheduleSycl(const Range & range, const OpenClKernel & kernel, const KernelArguments & args, Status & status)
    {
        UsmPointerStorage bufferStorage;

        status |= catchSyclExceptions([&]() mutable {
            ::sycl::kernel syclKernel = kernel.toSycl(_queue.get_context());

            auto event = _queue.submit([&](::sycl::handler & cgh) {
                passArguments(_queue, cgh, bufferStorage, args, status);
                DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
                cgh.parallel_for(range, syclKernel);
            });
            event.wait_and_throw();
        });
    }

    void passArguments(::sycl::queue & queue, ::sycl::handler & cgh, UsmPointerStorage & storage, const KernelArguments & args, Status & status) const
    {
        for (size_t i = 0; i < args.size(); i++)
        {
            const auto & arg = args.get(i);
            SyclKernelSchedulerArgHandler argHandler(queue, cgh, storage, i, arg);
            TypeDispatcher::dispatch(arg.dataType(), argHandler, status);
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
        }
    }

private:
    ::sycl::queue & _queue;
};

} // namespace interface1
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
/// \endcond

#endif
