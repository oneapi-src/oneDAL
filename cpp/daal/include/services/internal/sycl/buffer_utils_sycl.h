/* file: buffer_utils_sycl.h */
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
#ifdef DAAL_SYCL_INTERFACE_USM
    struct UsmDeleter
    {
        ::sycl::queue queue;

        explicit UsmDeleter(const ::sycl::queue & q) : queue(q) {}

        void operator()(const void * ptr) const { ::sycl::free(const_cast<void *>(ptr), queue); }
    };

    struct AllocateUSMBacked
    {
        const ::sycl::queue & queue;
        size_t bufferSize;
        UniversalBuffer buffer;

        explicit AllocateUSMBacked(const ::sycl::queue & q, size_t size) : queue(q), bufferSize(size) {}

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            T * usmPtr = ::sycl::malloc_device<T>(bufferSize, queue);
            if (usmPtr == nullptr)
            {
                status |= services::ErrorMemoryAllocationFailed;
                return;
            }
            services::SharedPtr<T> usmSharedPtr(usmPtr, UsmDeleter { queue });
            buffer = services::internal::Buffer<T>(usmSharedPtr, bufferSize, queue, status);
        }
    };

    static UniversalBuffer allocateUSMBacked(const ::sycl::queue & q, TypeId type, size_t bufferSize, Status & status)
    {
        AllocateUSMBacked allocateOp(q, bufferSize);
        TypeDispatcher::dispatch(type, allocateOp, status);
        return allocateOp.buffer;
    }
#endif

public:
    static UniversalBuffer allocate(const ::sycl::queue & q, TypeId type, size_t bufferSize, Status & status)
    {
#ifdef DAAL_SYCL_INTERFACE_USM
        return BufferAllocator::allocateUSMBacked(q, type, bufferSize, status);
#else
        static_assert(false, "Allocations of sycl buffers are no longer supported");
#endif // DAAL_SYCL_INTERFACE_USM
    }
};

class BufferCopier
{
private:
    struct Execute
    {
        ::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        size_t dstOffset;
        UniversalBuffer & srcUnivers;
        size_t srcOffset;
        size_t count;

        explicit Execute(::sycl::queue & queue, UniversalBuffer & dst, size_t desOffset, UniversalBuffer & src, size_t srcOffset, size_t count)
            : queue(queue), dstUnivers(dst), dstOffset(desOffset), srcUnivers(src), srcOffset(srcOffset), count(count)
        {}

#ifdef DAAL_SYCL_INTERFACE_USM
        template <typename T>
        Status copyOp(const Buffer<T> & srcBuffer, const Buffer<T> & dstBuffer)
        {
            using namespace ::sycl;

            Status status;
            auto src = srcBuffer.toUSM(queue, data_management::readOnly, status);
            DAAL_CHECK_STATUS_VAR(status);

            auto dst = dstBuffer.toUSM(queue, data_management::writeOnly, status);
            DAAL_CHECK_STATUS_VAR(status);

            auto * src_raw = src.get() + srcOffset;
            auto * dst_raw = dst.get() + dstOffset;

            const size_t bytes_count = sizeof(T) * count;
            DAAL_ASSERT(bytes_count >= count);

            return catchSyclExceptions([&]() mutable {
                auto event = queue.memcpy(dst_raw, src_raw, bytes_count);
                event.wait_and_throw();
            });
        }
#endif

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(srcUnivers, T);
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(dstUnivers, T);

            const auto & srcBuffer = srcUnivers.get<T>();
            const auto & dstBuffer = dstUnivers.get<T>();

            DAAL_ASSERT(srcBuffer.size() >= srcOffset + count);
            DAAL_ASSERT(dstBuffer.size() >= dstOffset + count);

#ifdef DAAL_SYCL_INTERFACE_USM
            status |= copyOp(srcBuffer, dstBuffer);
#else
            static_assert(false, "Support of USM memory is required to copy data in service::Buffer");
#endif
        }
    };

public:
    static void copy(::sycl::queue & queue, UniversalBuffer & dest, size_t dstOffset, UniversalBuffer & src, size_t srcOffset, size_t count,
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
        ::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        size_t dstOffset;
        void * srcArray;
        size_t srcCount;
        size_t srcOffset;
        size_t count;

        explicit Execute(::sycl::queue & queue, UniversalBuffer & dst, size_t desOffset, void * src, size_t srcCount, size_t srcOffset, size_t count)
            : queue(queue), dstUnivers(dst), dstOffset(desOffset), srcArray(src), srcCount(srcCount), srcOffset(srcOffset), count(count)
        {}

#ifdef DAAL_SYCL_INTERFACE_USM
        template <typename T>
        Status copyOp(const T * src, const Buffer<T> & dstBuffer)
        {
            using namespace ::sycl;

            Status status;

            auto sub = dstBuffer.getSubBuffer(dstOffset, count, status);
            DAAL_CHECK_STATUS_VAR(status);

            {
                // TODO: change to use toUSM() and queue.memcpy()
                auto dst = sub.toHost(data_management::writeOnly, status);
                DAAL_CHECK_STATUS_VAR(status);

                auto dst_raw = dst.get();

                const size_t size = sizeof(T) * count;
                DAAL_ASSERT(size >= count);

                int result = daal_memcpy_s(dst_raw, size, src, size);
                if (result)
                {
                    return services::ErrorMemoryCopyFailedInternal;
                }
            }
            return status;
        }
#endif

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(dstUnivers, T);

            auto src               = (T *)srcArray;
            const auto & dstBuffer = dstUnivers.get<T>();

            DAAL_ASSERT(srcArray);
            DAAL_ASSERT(srcCount >= srcOffset + count);
            DAAL_ASSERT(dstBuffer.size() >= dstOffset + count);

#ifdef DAAL_SYCL_INTERFACE_USM
            status |= copyOp(src, dstBuffer);
#else
            static_assert(false, "Support of USM memory is required to copy data in service::Buffer");
#endif
        }
    };

public:
    static void copy(::sycl::queue & queue, UniversalBuffer & dest, size_t dstOffset, void * src, size_t srcCount, size_t srcOffset, size_t count,
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
        ::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        double value;

        explicit Execute(::sycl::queue & queue, UniversalBuffer & dest, double value) : queue(queue), dstUnivers(dest), value(value) {}

#ifdef DAAL_SYCL_INTERFACE_USM
        template <typename T>
        Status fillOp(const Buffer<T> & dstBuffer)
        {
            Status status;
            auto dstPtr = dstBuffer.toUSM(queue, data_management::writeOnly, status);
            DAAL_CHECK_STATUS_VAR(status);

            return catchSyclExceptions([&]() mutable {
                auto event = queue.fill(dstPtr.get(), static_cast<T>(value), dstBuffer.size());
                event.wait_and_throw();
            });
        }
#endif

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(dstUnivers, T);

            const auto & dstBuffer = dstUnivers.get<T>();

#ifdef DAAL_SYCL_INTERFACE_USM
            status |= fillOp(dstBuffer);
#else
            static_assert(false, "Support of USM memory is required to fill data in service::Buffer");
#endif
        }
    };

public:
    static void fill(::sycl::queue & queue, UniversalBuffer & dest, double value, Status & status)
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
