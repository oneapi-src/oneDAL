/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/test/engine/communicator.hpp"

namespace oneapi::dal::test::engine {

template <typename Op>
inline void switch_by_dtype(const data_type& dtype, const Op& op) {
    switch (dtype) {
        case data_type::int8: return op(std::int8_t{});
        case data_type::uint8: return op(std::uint8_t{});
        case data_type::int16: return op(std::int16_t{});
        case data_type::uint16: return op(std::uint16_t{});
        case data_type::int32: return op(std::int32_t{});
        case data_type::uint32: return op(std::uint32_t{});
        case data_type::int64: return op(std::int64_t{});
        case data_type::uint64: return op(std::uint64_t{});
        case data_type::float32: return op(float{});
        case data_type::float64: return op(double{});
        default:
            throw std::runtime_error{
                "Thread communicator does not support reduction for given types"
            };
    }
}

template <typename T>
struct reduce_op_sum {
    void operator()(const T* src, T* dst, std::int64_t count) const {
        for (std::int64_t i = 0; i < count; i++) {
            dst[i] += src[i];
        }
    }
};

template <typename T, typename Op>
inline void switch_by_reduce_op(const dal::detail::spmd_reduce_op& reduce_op_id, const Op& op) {
    using dal::detail::spmd_reduce_op;

    switch (reduce_op_id) {
        case spmd_reduce_op::sum: return op(reduce_op_sum<T>{});
        default:
            throw std::runtime_error{
                "Thread communicator does not support reduction for given operation"
            };
    }
}

void thread_communicator_allreduce::reduce(const byte_t* src,
                                           byte_t* dst,
                                           std::int64_t count,
                                           const data_type& dtype,
                                           const dal::detail::spmd_reduce_op& op_id) {
    switch_by_dtype(dtype, [&](auto _) {
        using value_t = decltype(_);
        reduce_impl<value_t>(src, dst, count, op_id);
    });
}

void thread_communicator_allreduce::fill_with_zeros(byte_t* dst,
                                                    std::int64_t count,
                                                    const data_type& dtype) {
    switch_by_dtype(dtype, [&](auto _) {
        using value_t = decltype(_);
        fill_with_zeros_impl<value_t>(dst, count);
    });
}

template <typename T>
void thread_communicator_allreduce::reduce_impl(const byte_t* src_bytes,
                                                byte_t* dst_bytes,
                                                std::int64_t count_in_bytes,
                                                const dal::detail::spmd_reduce_op& op_id) {
    ONEDAL_ASSERT(src_bytes);
    ONEDAL_ASSERT(dst_bytes);
    ONEDAL_ASSERT(count_in_bytes % sizeof(T) == 0);

    const std::int64_t count = count_in_bytes / sizeof(T);
    const T* src = reinterpret_cast<const T*>(src_bytes);
    T* dst = reinterpret_cast<T*>(dst_bytes);

    switch_by_reduce_op<T>(op_id, [&](auto reduce_op) {
        reduce_op(src, dst, count);
    });
}

template <typename T>
void thread_communicator_allreduce::fill_with_zeros_impl(byte_t* dst_bytes,
                                                         std::int64_t count_in_bytes) {
    ONEDAL_ASSERT(dst_bytes);
    ONEDAL_ASSERT(count_in_bytes % sizeof(T) == 0);

    const std::int64_t count = count_in_bytes / sizeof(T);
    T* dst = reinterpret_cast<T*>(dst_bytes);

    for (std::int64_t i = 0; i < count; i++) {
        dst[i] = T(0);
    }
}

} // namespace oneapi::dal::test::engine
