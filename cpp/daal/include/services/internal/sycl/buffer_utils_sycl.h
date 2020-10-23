/* file: buffer_utils_sycl.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_BUFFER_UTILS_SYCL_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_BUFFER_UTILS_SYCL_H__

#ifndef DAAL_SYCL_INTERFACE
    #error "DAAL_SYCL_INTERFACE must be defined to include this file"
#endif

#include "services/internal/sycl/types_utils.h"

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
            status |= catchSyclExceptions([&]() mutable { buffer = Buffer<T>(cl::sycl::buffer<T, 1>(bufferSize), status); });
        }
    };

public:
    static UniversalBuffer allocate(TypeId type, size_t bufferSize, Status & status)
    {
        Allocate allocateOp(bufferSize);
        TypeDispatcher::dispatch(type, allocateOp, status);
        return allocateOp.buffer;
    }
};

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
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(srcUnivers, T);
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(dstUnivers, T);

            auto src = srcUnivers.get<T>().toSycl(status);
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

            auto dst = dstUnivers.get<T>().toSycl(status);
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

            DAAL_ASSERT(src.get_count() >= srcOffset + count);
            DAAL_ASSERT(dst.get_count() >= dstOffset + count);

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
                     Status & status)
    {
        DAAL_ASSERT(!src.empty());
        DAAL_ASSERT(!dest.empty());
        DAAL_ASSERT(src.type() == dest.type());

        Execute op(queue, dest, dstOffset, src, srcOffset, count);
        TypeDispatcher::dispatch(dest.type(), op, status);
    }
};

class ArrayCopier
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        size_t dstOffset;
        void * srcArray;
        size_t srcCount;
        size_t srcOffset;
        size_t count;

        explicit Execute(cl::sycl::queue & queue, UniversalBuffer & dst, size_t desOffset, void * src, size_t srcCount, size_t srcOffset,
                         size_t count)
            : queue(queue), dstUnivers(dst), dstOffset(desOffset), srcArray(src), srcCount(srcCount), srcOffset(srcOffset), count(count)
        {}

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(dstUnivers, T);

            auto src = (T *)srcArray;
            auto dst = dstUnivers.get<T>().toSycl(status);
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

            DAAL_ASSERT(srcArray);
            DAAL_ASSERT(srcCount >= srcOffset + count);
            DAAL_ASSERT(dst.get_count() >= dstOffset + count);

            status |= catchSyclExceptions([&]() mutable {
                cl::sycl::event event = queue.submit([&](cl::sycl::handler & cgh) {
                    auto dst_acc = dst.template get_access<cl::sycl::access::mode::write>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(dstOffset));
                    cgh.copy(src + srcOffset, dst_acc);
                });
                event.wait_and_throw();
            });
        }
    };

public:
    static void copy(cl::sycl::queue & queue, UniversalBuffer & dest, size_t dstOffset, void * src, size_t srcCount, size_t srcOffset, size_t count,
                     Status & status)
    {
        DAAL_ASSERT(!dest.empty());

        Execute op(queue, dest, dstOffset, src, srcCount, srcOffset, count);
        TypeDispatcher::dispatch(dest.type(), op, status);
    }
};

class BufferFiller
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        double value;

        explicit Execute(cl::sycl::queue & queue, UniversalBuffer & dest, double value) : queue(queue), dstUnivers(dest), value(value) {}

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(dstUnivers, T);

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
    static void fill(cl::sycl::queue & queue, UniversalBuffer & dest, double value, Status & status)
    {
        DAAL_ASSERT(!dest.empty());

        Execute op(queue, dest, value);
        TypeDispatcher::dispatch(dest.type(), op, status);
    }
};
} // namespace interface1

using interface1::BufferAllocator;
using interface1::BufferCopier;
using interface1::BufferFiller;

} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
/// \endcond

#endif
