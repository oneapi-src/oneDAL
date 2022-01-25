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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_EXECUTION_CONTEXT_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_EXECUTION_CONTEXT_H__

#include "services/error_handling.h"
#include "services/internal/any.h"
#include "services/internal/buffer.h"
#include "services/internal/sycl/types.h"
#include "services/internal/sycl/math/types.h"

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
/** @ingroup oneapi_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__KERNELRANGE"></a>
 *  \brief Local range of computations run in parallel
 */
class KernelRange : public Base
{
public:
    KernelRange() : _upper1(0), _upper2(0), _upper3(0), _dimensions(0) {}

    explicit KernelRange(size_t upper) : _upper1(upper), _upper2(1), _upper3(1), _dimensions(1) {}

    explicit KernelRange(size_t upper1, size_t upper2) : _upper1(upper1), _upper2(upper2), _upper3(1), _dimensions(2) {}

    explicit KernelRange(size_t upper1, size_t upper2, size_t upper3) : _upper1(upper1), _upper2(upper2), _upper3(upper3), _dimensions(3) {}

    size_t upper1() const { return _upper1; }

    size_t upper2() const { return _upper2; }

    size_t upper3() const { return _upper3; }

    size_t dimensions() const { return _dimensions; }

private:
    size_t _upper1;
    size_t _upper2;
    size_t _upper3;

    size_t _dimensions;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__KERNELNDRANGE"></a>
 *  \brief Class containing local and global ranges
 */
class KernelNDRange : public Base
{
public:
    explicit KernelNDRange(size_t dimensions) : _dimensions(dimensions) {}

    void global(const KernelRange & range, Status & st)
    {
        if (_dimensions != range.dimensions())
        {
            st |= ErrorIncorrectParameter;
            return;
        }
        _globalRange = range;
    }

    void local(const KernelRange & range, Status & st)
    {
        if (_dimensions != range.dimensions())
        {
            st |= ErrorIncorrectParameter;
            return;
        }
        _localRange = range;
    }

    const KernelRange & local() { return _localRange; }

    const KernelRange & global() const { return _globalRange; }

    const KernelRange & local() const { return _localRange; }

    size_t dimensions() const { return _dimensions; }

private:
    KernelRange _globalRange;
    KernelRange _localRange;
    size_t _dimensions;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__KERNELARGUMENTTYPES"></a>
 *  \brief Types of arguments can be passed into kernel
 */
class KernelArgumentTypes
{
public:
    enum Type
    {
        publicBuffer,
        publicConstant,
        privateBuffer
    };

private:
    KernelArgumentTypes();
};
typedef KernelArgumentTypes::Type KernelArgumentType;

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__KERNELARGUMENT"></a>
 *  \brief Container for argument to be passed into kernel
 */
class KernelArgument : public Base
{
public:
    KernelArgument() : _dataType(TypeIds::custom), _argType(KernelArgumentTypes::publicConstant), _accessMode(AccessModeIds::read) {}

    template <typename T>
    void set(const T & value)
    {
        _dataType   = TypeIds::id<T>();
        _value      = value;
        _argType    = KernelArgumentTypes::publicConstant;
        _accessMode = AccessModeIds::read;
    }

    template <typename T>
    void set(const Buffer<T> & buffer, AccessModeId accessMode = AccessModeIds::read)
    {
        _dataType   = TypeIds::id<T>();
        _value      = buffer;
        _argType    = KernelArgumentTypes::publicBuffer;
        _accessMode = accessMode;
    }

    void set(const UniversalBuffer & buffer, AccessModeId accessMode = AccessModeIds::read)
    {
        _dataType   = buffer.type();
        _value      = buffer.any();
        _argType    = KernelArgumentTypes::publicBuffer;
        _accessMode = accessMode;
    }

    void set(const LocalBuffer & buffer)
    {
        _dataType   = buffer.type();
        _value      = buffer;
        _argType    = KernelArgumentTypes::privateBuffer;
        _accessMode = AccessModeIds::readwrite;
    }

    TypeId dataType() const { return _dataType; }

    AccessModeId accessMode() const { return _accessMode; }

    KernelArgumentType argType() const { return _argType; }

    template <typename T>
    const T & get() const
    {
        DAAL_ASSERT(_value.check<T>());
        return _value.get<T>();
    }

private:
    TypeId _dataType;
    KernelArgumentType _argType;
    AccessModeId _accessMode;
    Any _value;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__KERNELARGUMENTS"></a>
 *  \brief Container for all kernel arguments
 */
class KernelArguments : public Base
{
public:
    KernelArguments() {}

    explicit KernelArguments(size_t argsNum, Status & status) : _args(argsNum)
    {
        DAAL_CHECK_COND_ERROR(_args.data(), status, ErrorMemoryAllocationFailed);
    }

    template <typename T>
    void set(size_t index, const T & value)
    {
        _args[index].set(value);
    }

    template <typename T>
    void set(size_t index, const Buffer<T> & buffer, AccessModeId accessMode = AccessModeIds::read)
    {
        _args[index].set(buffer, accessMode);
    }

    void set(size_t index, const UniversalBuffer & buffer, AccessModeId accessMode = AccessModeIds::read) { _args[index].set(buffer, accessMode); }

    void set(size_t index, const LocalBuffer & buffer) { _args[index].set(buffer); }

    const KernelArgument & get(size_t index) const { return _args[index]; }

    size_t size() const { return _args.size(); }

private:
    /* Disable copy & assignment */
    KernelArguments(const KernelArguments &);
    KernelArguments & operator=(const KernelArguments &);

    Collection<KernelArgument> _args;
};

/* Forward declarations of possible kernel types */
class OpenClKernel;

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__KERNELSCHEDULERIFACE"></a>
 *  \brief Interface for pushing kernel to run
 */
class KernelSchedulerIface
{
public:
    virtual ~KernelSchedulerIface() {}

    virtual void schedule(const OpenClKernel & kernel, const KernelRange & range, const KernelArguments & args, Status & st) = 0;

    virtual void schedule(const OpenClKernel & kernel, const KernelNDRange & range, const KernelArguments & args, Status & st) = 0;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__KERNELIFACE"></a>
 *  \brief Interface for kernel
 */
class KernelIface
{
public:
    virtual ~KernelIface() {}

    virtual void schedule(KernelSchedulerIface & scheduler, const KernelRange & range, const KernelArguments & args, Status & st) const = 0;

    virtual void schedule(KernelSchedulerIface & scheduler, const KernelNDRange & range, const KernelArguments & args, Status & st) const = 0;
};
typedef SharedPtr<KernelIface> KernelPtr;

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__CLKERNELFACTORYIFACE"></a>
 *  \brief Interface for factory of kernels
 */
class ClKernelFactoryIface
{
public:
    virtual ~ClKernelFactoryIface() {}
    virtual void build(ExecutionTargetId target, const char * key, const char * program, const char * options, Status & st) = 0;
    virtual SharedPtr<KernelIface> getKernel(const char * kernelName, Status & st)                                          = 0;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__INFODEVICE"></a>
 *  \brief Struct containing device information
 */
struct InfoDevice
{
    size_t maxWorkGroupSize;
    bool isCpu;
    size_t maxMemAllocSize;
    size_t globalMemSize;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__EXECUTIONCONTEXTIFACE"></a>
 *  \brief Interface of execution context
 */
class ExecutionContextIface
{
public:
    virtual ~ExecutionContextIface() {}

    virtual void run(const KernelRange & range, const KernelPtr & kernel, const KernelArguments & args, Status & st) = 0;

    virtual void run(const KernelNDRange & range, const KernelPtr & kernel, const KernelArguments & args, Status & st) = 0;

    virtual void gemm(math::Transpose transa, math::Transpose transb, size_t m, size_t n, size_t k, double alpha, const UniversalBuffer & a_buffer,
                      size_t lda, size_t offsetA, const UniversalBuffer & b_buffer, size_t ldb, size_t offsetB, double beta,
                      UniversalBuffer & c_buffer, size_t ldc, size_t offsetC, Status & st) = 0;

    virtual void syrk(math::UpLo upper_lower, math::Transpose trans, size_t n, size_t k, double alpha, const UniversalBuffer & a_buffer, size_t lda,
                      size_t offsetA, double beta, UniversalBuffer & c_buffer, size_t ldc, size_t offsetC, Status & st) = 0;

    virtual void axpy(const uint32_t n, const double a, const UniversalBuffer x_buffer, const int incx, const UniversalBuffer y_buffer,
                      const int incy, Status & st) = 0;

    virtual void potrf(math::UpLo uplo, size_t n, UniversalBuffer & a_buffer, size_t lda, Status & st) = 0;

    virtual void potrs(math::UpLo uplo, size_t n, size_t ny, UniversalBuffer & a_buffer, size_t lda, UniversalBuffer & b_buffer, size_t ldb,
                       Status & st) = 0;

    virtual void copy(UniversalBuffer dest, size_t desOffset, UniversalBuffer src, size_t srcOffset, size_t count, Status & st) = 0;

    virtual void fill(UniversalBuffer dest, double value, Status & st) = 0;

    virtual UniversalBuffer allocate(TypeId type, size_t bufferSize, Status & st) = 0;

    virtual ClKernelFactoryIface & getClKernelFactory() = 0;

    virtual InfoDevice & getInfoDevice() = 0;

    virtual void copy(UniversalBuffer dest, size_t desOffset, void * src, size_t srcCount, size_t srcOffset, size_t count, Status & st) = 0;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__CPUKERNELFACTORY"></a>
 *  \brief Factory for host-only kernels
 */
class CpuKernelFactory : public Base, public ClKernelFactoryIface
{
public:
    virtual void build(ExecutionTargetId /*target*/, const char * /*key*/, const char * /*program*/, const char * /*options = ""*/,
                       Status & /*status*/) DAAL_C11_OVERRIDE
    {}
    virtual SharedPtr<KernelIface> getKernel(const char * /*kernelName*/, Status & /*status*/) DAAL_C11_OVERRIDE { return SharedPtr<KernelIface>(); }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__CPUEXECUTIONCONTEXTIMPL"></a>
 *  \brief Host execution context
 */
class CpuExecutionContextImpl : public Base, public ExecutionContextIface
{
public:
    CpuExecutionContextImpl()
    {
        _infoDevice.isCpu            = true;
        _infoDevice.maxWorkGroupSize = 0;
        _infoDevice.maxMemAllocSize  = 0;
        _infoDevice.globalMemSize    = 0;
    }

    void run(const KernelRange & /*range*/, const KernelPtr & /*kernel*/, const KernelArguments & /*args*/, Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
    }

    void run(const KernelNDRange & /*range*/, const KernelPtr & /*kernel*/, const KernelArguments & /*args*/, Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
    }

    void gemm(math::Transpose /*transa*/, math::Transpose /*transb*/, size_t /*m*/, size_t /*n*/, size_t /*k*/, double /*alpha*/,
              const UniversalBuffer & /*a_buffer*/, size_t /*lda*/, size_t /*offsetA*/, const UniversalBuffer & /*b_buffer*/, size_t /*ldb*/,
              size_t /*offsetB*/, double /*beta*/, UniversalBuffer & /*c_buffer*/, size_t /*ldc*/, size_t /*offsetC*/, Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
    }

    void syrk(math::UpLo /*upper_lower*/, math::Transpose /*trans*/, size_t /*n*/, size_t /*k*/, double /*alpha*/,
              const UniversalBuffer & /*a_buffer*/, size_t /*lda*/, size_t /*offsetA*/, double /*beta*/, UniversalBuffer & /*c_buffer*/,
              size_t /*ldc*/, size_t /*offsetC*/, Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
    }

    void axpy(const uint32_t /*n*/, const double /*a*/, const UniversalBuffer /*x_buffer*/, const int /*incx*/, const UniversalBuffer /*y_buffer*/,
              const int /*incy*/, Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
    }

    void potrf(math::UpLo /*uplo*/, size_t /*n*/, UniversalBuffer & /*a_buffer*/, size_t /*lda*/, Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
    }

    void potrs(math::UpLo /*uplo*/, size_t /*n*/, size_t /*ny*/, UniversalBuffer & /*a_buffer*/, size_t /*lda*/, UniversalBuffer & /*b_buffer*/,
               size_t /*ldb*/, Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
    }

    void copy(UniversalBuffer /*dest*/, size_t /*desOffset*/, UniversalBuffer /*src*/, size_t /*srcOffset*/, size_t /*count*/,
              Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
    }

    void fill(UniversalBuffer /*dest*/, double /*value*/, Status & st) DAAL_C11_OVERRIDE { st |= ErrorMethodNotImplemented; }

    UniversalBuffer allocate(TypeId /*type*/, size_t /*bufferSize*/, Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
        return UniversalBuffer();
    }

    ClKernelFactoryIface & getClKernelFactory() DAAL_C11_OVERRIDE { return _factory; }

    InfoDevice & getInfoDevice() DAAL_C11_OVERRIDE { return _infoDevice; }

    void copy(UniversalBuffer /*dest*/, size_t /*desOffset*/, void * /*src*/, size_t /*srcCount*/, size_t /*srcOffset*/, size_t /*count*/,
              Status & st) DAAL_C11_OVERRIDE
    {
        st |= ErrorMethodNotImplemented;
    }

private:
    CpuKernelFactory _factory;
    InfoDevice _infoDevice;
};

/** @} */
} // namespace interface1

using interface1::KernelRange;
using interface1::KernelNDRange;
using interface1::KernelArgumentTypes;
using interface1::KernelArgumentType;
using interface1::KernelArgument;
using interface1::KernelArguments;
using interface1::UniversalBuffer;
using interface1::KernelSchedulerIface;
using interface1::KernelIface;
using interface1::KernelPtr;
using interface1::ClKernelFactoryIface;
using interface1::InfoDevice;
using interface1::ExecutionContextIface;
using interface1::CpuKernelFactory;
using interface1::CpuExecutionContextImpl;

} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
