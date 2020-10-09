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

#ifdef DAAL_SYCL_INTERFACE
    #ifndef __DAAL_SERVICES_INTERNAL_SYCL_TYPES_UTILS_CXX11_H__
        #define __DAAL_SERVICES_INTERNAL_SYCL_TYPES_UTILS_CXX11_H__

        #include <CL/sycl.hpp>

        #include "services/internal/sycl/types_utils.h"
        #include "services/daal_memory.h"

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
    struct UsmDeleter
    {
        cl::sycl::queue queue;

        explicit UsmDeleter(const cl::sycl::queue & q) : queue(q) {}

        void operator()(const void * ptr) const { cl::sycl::free(const_cast<void *>(ptr), queue); }
    };

    struct AllocateVanilla
    {
        UniversalBuffer buffer;
        size_t bufferSize;

        explicit AllocateVanilla(size_t size) : bufferSize(size) {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            buffer = services::internal::Buffer<T>(cl::sycl::buffer<T, 1>(bufferSize));
        }
    };

    struct AllocateUSMBacked
    {
        const cl::sycl::queue & queue;
        size_t bufferSize;
        UniversalBuffer buffer;

        explicit AllocateUSMBacked(const cl::sycl::queue & q, size_t size) : queue(q), bufferSize(size) {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            const auto usmKind = cl::sycl::usm::alloc::shared;
            T * usmPtr         = cl::sycl::malloc<T>(bufferSize, queue, usmKind); //TODO: handle memory allocation error
            services::SharedPtr<T> usmSharedPtr(usmPtr, UsmDeleter { queue });
            buffer = services::internal::Buffer<T>(usmSharedPtr, bufferSize, usmKind);
        }
    };

public:
    static UniversalBuffer allocateVanilla(TypeId type, size_t bufferSize)
    {
        AllocateVanilla allocateOp(bufferSize);
        TypeDispatcher::dispatch(type, allocateOp);
        return allocateOp.buffer;
    }

    static UniversalBuffer allocateUSMBacked(const cl::sycl::queue & q, TypeId type, size_t bufferSize)
    {
        AllocateUSMBacked allocateOp(q, bufferSize);
        TypeDispatcher::dispatch(type, allocateOp);
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
        void operator()(Typelist<T>)
        {
            const auto & src_buffer = srcUnivers.get<T>();
            const auto & dst_buffer = dstUnivers.get<T>();

            if (src_buffer.isUSMBacked() && dst_buffer.isUSMBacked())
            {
                auto src = src_buffer.toUSM();
                auto dst = dst_buffer.toUSM();

                auto src_raw = src.get() + srcOffset;
                auto dst_raw = dst.get() + dstOffset;

                queue.memcpy(dst_raw, src_raw, sizeof(T) * count).wait();
            }
            else if (src_buffer.isUSMBacked())
            {
                // this branch is a workaround on the SYCL RT bug
                // on copy operation from usm-backed buffer to the vanilla buffer
                auto src   = src_buffer.toUSM();
                auto dst   = dst_buffer.toSycl();
                auto event = queue.submit([&](cl::sycl::handler & cgh) {
                    auto dst_acc = dst.template get_access<cl::sycl::access::mode::write>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(dstOffset));
                    cgh.copy(src.get() + srcOffset, dst_acc);
                });
                event.wait();
            }
            else
            {
                auto src   = src_buffer.toSycl();
                auto dst   = dst_buffer.toSycl();
                auto event = queue.submit([&](cl::sycl::handler & cgh) {
                    auto src_acc = src.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(srcOffset));
                    auto dst_acc = dst.template get_access<cl::sycl::access::mode::write>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(dstOffset));
                    cgh.copy(src_acc, dst_acc);
                });
                event.wait();
            }
        }
    };

public:
    static void copy(cl::sycl::queue & queue, UniversalBuffer & dest, size_t dstOffset, UniversalBuffer & src, size_t srcOffset, size_t count)
    {
        Execute op(queue, dest, dstOffset, src, srcOffset, count);
        TypeDispatcher::dispatch(dest.type(), op);
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
        void operator()(Typelist<T>)
        {
            auto src                = (T *)srcArray;
            const auto & dst_buffer = dstUnivers.get<T>();

            if (dst_buffer.isUSMBacked())
            {
                auto dst     = dst_buffer.toUSM();
                auto dst_raw = dst.get() + dstOffset;
                DAAL_ASSERT(((cl::sycl::get_pointer_type(dst_raw, queue.get_context()) == cl::sycl::usm::alloc::shared)
                             || (cl::sycl::get_pointer_type(dst_raw, queue.get_context()) == cl::sycl::usm::alloc::host)));

                const size_t size = sizeof(T) * count;
                daal_memcpy_s(dst_raw, size, src + srcOffset, size); // TODO: check return status!
            }
            else
            {
                auto dst              = dst_buffer.toSycl();
                cl::sycl::event event = queue.submit([&](cl::sycl::handler & cgh) {
                    auto dst_acc = dst.template get_access<cl::sycl::access::mode::write>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(dstOffset));
                    cgh.copy(src + srcOffset, dst_acc);
                });
                event.wait();
            }
        }
    };

public:
    static void copy(cl::sycl::queue & queue, UniversalBuffer & dest, size_t dstOffset, void * src, size_t srcOffset, size_t count)
    {
        Execute op(queue, dest, dstOffset, src, srcOffset, count);
        TypeDispatcher::dispatch(dest.type(), op);
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

        explicit Execute(cl::sycl::queue & queue, UniversalBuffer & dest, double value) : queue(queue), dstUnivers(dest), value(value) {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            const auto & dstBuffer = dstUnivers.get<T>();

            if (dstBuffer.isUSMBacked())
            {
                auto dstPtr = dstBuffer.toUSM();
                queue.fill(dstPtr.get(), static_cast<T>(value), dstBuffer.size()).wait();
            }
            else
            {
                auto dst              = dstBuffer.toSycl();
                cl::sycl::event event = queue.submit([&](cl::sycl::handler & cgh) {
                    auto acc = dst.template get_access<cl::sycl::access::mode::write>(cgh);
                    cgh.fill(acc, static_cast<T>(value));
                });
                event.wait();
            }
        }
    };

public:
    static void fill(cl::sycl::queue & queue, UniversalBuffer & dest, double value)
    {
        Execute op(queue, dest, value);
        TypeDispatcher::dispatch(dest.type(), op);
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
#endif // DAAL_SYCL_INTERFACE
