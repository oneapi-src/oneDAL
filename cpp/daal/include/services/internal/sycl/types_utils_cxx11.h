/* file: types_utils_cxx11.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_TYPES_UTILS_CXX11_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_TYPES_UTILS_CXX11_H__

#include "services/internal/sycl/types_utils.h"

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
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERALLOCATOR"></a>
 *  \brief Allocator for UniversalBuffer
 */
class BufferAllocator
{
private:
    struct Allocate
    {
        UniversalBuffer buffer;
        size_t bufferSize;

        explicit Allocate(size_t size) : bufferSize(size) {}

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            status |= catchSyclExceptions([&]() mutable {
                buffer = services::internal::Buffer<T>(cl::sycl::buffer<T, 1>(bufferSize), status);
            });
        }
    };

public:
    static UniversalBuffer allocate(TypeId type, size_t bufferSize, Status & status)
    {
        Allocate allocateOp(bufferSize);
        TypeDispatcher::dispatch(type, allocateOp, status);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, UniversalBuffer());
        return allocateOp.buffer;
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERCOPIER"></a>
 *  \brief Copier for two UniversalBuffers
 */
class BufferCopier
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        size_t dstOffset;
        UniversalBuffer & srcUnivers;
        size_t srcOffset;
        size_t count;

        explicit Execute(cl::sycl::queue & queue, UniversalBuffer & dst, size_t desOffset, UniversalBuffer & src, size_t srcOffset, size_t count)
            : queue(queue), dstUnivers(dst), dstOffset(desOffset), srcUnivers(src), srcOffset(srcOffset), count(count)
        {}

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            auto src = srcUnivers.get<T>().toSycl(status);
            DAAL_CHECK_STATUS_VAR(status);

            auto dst = dstUnivers.get<T>().toSycl(status);
            DAAL_CHECK_STATUS_VAR(status);

            status |= catchSyclExceptions([&]() mutable {
                cl::sycl::event event = queue.submit([&](cl::sycl::handler & cgh) {
                    auto src_acc = src.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(srcOffset));
                    auto dst_acc = dst.template get_access<cl::sycl::access::mode::write>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(dstOffset));
                    cgh.copy(src_acc, dst_acc);
                });
                event.wait_and_throw();
            });
        }
    };

public:
    static void copy(cl::sycl::queue & queue, UniversalBuffer & dest, size_t dstOffset, UniversalBuffer & src, size_t srcOffset, size_t count,
                     services::Status & status)
    {
        Execute op(queue, dest, dstOffset, src, srcOffset, count);
        TypeDispatcher::dispatch(dest.type(), op, status);
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__ARRAYCOPIER"></a>
 *  \brief Copier from array to UniversalBuffers
 */
class ArrayCopier
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        size_t dstOffset;
        void * srcArray;
        size_t srcOffset;
        size_t count;

        explicit Execute(cl::sycl::queue & queue, UniversalBuffer & dst, size_t desOffset, void * src, size_t srcOffset, size_t count)
            : queue(queue), dstUnivers(dst), dstOffset(desOffset), srcArray(src), srcOffset(srcOffset), count(count)
        {}

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            auto src = (T *)srcArray;
            auto dst = dstUnivers.get<T>().toSycl(status);
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

            status |= catchSyclExceptions([&]() mutable {
                cl::sycl::event event = queue.submit([&](cl::sycl::handler & cgh) {
                    auto dst_acc = dst.template get_access<cl::sycl::access::mode::write>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(dstOffset));
                    cgh.copy(src, dst_acc);
                });
                event.wait_and_throw();
            });
        }
    };

public:
    static void copy(cl::sycl::queue & queue, UniversalBuffer & dest, size_t dstOffset, void * src, size_t srcOffset, size_t count,
                     services::Status & status)
    {
        Execute op(queue, dest, dstOffset, src, srcOffset, count);
        TypeDispatcher::dispatch(dest.type(), op, status);
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERFILLER"></a>
 *  \brief Fills UniversalBuffers with single value
 */
class BufferFiller
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        double value;

        explicit Execute(cl::sycl::queue & queue, UniversalBuffer & dest, double value)
            : queue(queue), dstUnivers(dest), value(value)
        {}

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            auto dst = dstUnivers.get<T>().toSycl(status);
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

            status |= catchSyclExceptions([&]() mutable {
                cl::sycl::event event = queue.submit([&](cl::sycl::handler & cgh) {
                    auto acc = dst.template get_access<cl::sycl::access::mode::write>(cgh);
                    cgh.fill(acc, static_cast<T>(value));
                });
                event.wait_and_throw();
            });
        }
    };

public:
    static void fill(cl::sycl::queue & queue, UniversalBuffer & dest, double value, services::Status & status)
    {
        Execute op(queue, dest, value);
        TypeDispatcher::dispatch(dest.type(), op, status);
    }
};

/** @} */
} // namespace interface1

using interface1::BufferAllocator;
using interface1::BufferCopier;
using interface1::BufferFiller;

} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
