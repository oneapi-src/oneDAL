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

#include "oneapi/dal/test/engine/thread_communicator.hpp"

namespace oneapi::dal::test::engine {

void thread_communicator_context::init(std::int64_t rank) {
    std::unique_lock<std::mutex> lock(internal_lock_);

    thread_id_map_[std::this_thread::get_id()] = rank;
    if (thread_id_map_.size() == std::size_t(thread_count_)) {
        cv_.notify_all();
    }
    else {
        cv_.wait(lock, [this]() {
            return thread_id_map_.size() == std::size_t(thread_count_);
        });
    }
}

void thread_communicator_bcast::operator()(byte_t* send_buf,
                                           std::int64_t count,
                                           const data_type& dtype,
                                           std::int64_t root) {
    ONEDAL_ASSERT(root >= 0);
    if (count == 0) {
        return;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(count > 0);

    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t size = dal::detail::check_mul_overflow(dtype_size, count);

    if (ctx_.get_this_thread_rank() == root) {
        source_count_ = count;
        source_buf_ = send_buf;
    }

    barrier_();

    if (ctx_.get_this_thread_rank() != root) {
        ONEDAL_ASSERT(source_buf_);
        ONEDAL_ASSERT(source_count_ > 0);
        ONEDAL_ASSERT(count <= source_count_);

        dal::detail::memcpy(dal::detail::default_host_policy{}, send_buf, source_buf_, size);
    }

    barrier_([&]() {
        source_count_ = 0;
        source_buf_ = nullptr;
    });
}

void thread_communicator_gather::operator()(const byte_t* send_buf,
                                            std::int64_t send_count,
                                            byte_t* recv_buf,
                                            std::int64_t recv_count,
                                            const data_type& dtype,
                                            std::int64_t root) {
    ONEDAL_ASSERT(root >= 0);
    if (send_count == 0) {
        return;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(send_count > 0);

    const std::int64_t rank = ctx_.get_this_thread_rank();
    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t send_size = dal::detail::check_mul_overflow(dtype_size, send_count);

    if (rank == root) {
        ONEDAL_ASSERT(recv_buf);
        ONEDAL_ASSERT(recv_count > 0);
        recv_count_ = recv_count;
        recv_buf_ = recv_buf;
    }

    barrier_();

    ONEDAL_ASSERT(recv_buf_);
    ONEDAL_ASSERT(recv_count_ > 0);
    ONEDAL_ASSERT(send_count <= recv_count_);

    const std::int64_t recv_size = dal::detail::check_mul_overflow(dtype_size, recv_count_);
    const std::int64_t offset = dal::detail::check_mul_overflow(rank, recv_size);

    for (std::int64_t i = 0; i < send_size; i++) {
        recv_buf_[offset + i] = send_buf[i];
    }

    barrier_([&]() {
        recv_count_ = 0;
        recv_buf_ = nullptr;
    });
}

void thread_communicator_gatherv::operator()(const byte_t* send_buf,
                                             std::int64_t send_count,
                                             byte_t* recv_buf,
                                             const std::int64_t* recv_counts,
                                             const std::int64_t* displs,
                                             const data_type& dtype,
                                             std::int64_t root) {
    ONEDAL_ASSERT(root >= 0);

    if (send_count == 0) {
        return;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(send_count > 0);

    const std::int64_t rank = ctx_.get_this_thread_rank();
    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t send_size = dal::detail::check_mul_overflow(dtype_size, send_count);

    if (rank == root) {
        ONEDAL_ASSERT(recv_buf);
        ONEDAL_ASSERT(displs);
        ONEDAL_ASSERT(recv_counts);
        recv_counts_ = recv_counts;
        displs_ = displs;
        recv_buf_ = recv_buf;
    }

    barrier_();

    ONEDAL_ASSERT(recv_counts_);
    ONEDAL_ASSERT(displs_);
    ONEDAL_ASSERT(recv_buf_);
    ONEDAL_ASSERT(send_count <= recv_counts_[rank]);

    const std::int64_t offset = dal::detail::check_mul_overflow(dtype_size, displs_[rank]);
    for (std::int64_t i = 0; i < send_size; i++) {
        recv_buf_[offset + i] = send_buf[i];
    }

    barrier_([&]() {
        recv_counts_ = nullptr;
        displs_ = nullptr;
        recv_buf_ = nullptr;
    });
}

void thread_communicator_allgather::operator()(const byte_t* send_buf,
                                               std::int64_t send_count,
                                               byte_t* recv_buf,
                                               std::int64_t recv_count,
                                               const data_type& dtype) {
    if (send_count == 0) {
        return;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(send_count > 0);
    ONEDAL_ASSERT(recv_buf);
    ONEDAL_ASSERT(recv_count > 0);
    ONEDAL_ASSERT(send_count == recv_count);

    const std::int64_t rank = ctx_.get_this_thread_rank();
    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t recv_size = dal::detail::check_mul_overflow(dtype_size, recv_count);

    send_buffers_[rank] = buffer_info{ send_buf, send_count };

    barrier_();

#ifdef ONEDAL_ENABLE_ASSERT
    if (rank == ctx_.get_root_rank()) {
        for (const auto& info : send_buffers_) {
            ONEDAL_ASSERT(info.count == send_count);
            ONEDAL_ASSERT(info.buf != nullptr);
        }
    }
#endif

    {
        std::int64_t r = 0;
        for ([[maybe_unused]] const auto& send : send_buffers_) {
            for (std::int64_t i = 0; i < recv_size; i++) {
                recv_buf[r * recv_size + i] = send.buf[i];
            }
            r++;
        }
    }

    barrier_([&]() {
        send_buffers_.clear();
        send_buffers_.resize(ctx_.get_thread_count());
    });
}

void thread_communicator_allreduce::operator()(const byte_t* send_buf,
                                               byte_t* recv_buf,
                                               std::int64_t count,
                                               const data_type& dtype,
                                               const dal::detail::spmd_reduce_op& op) {
    if (count == 0) {
        return;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(recv_buf);
    ONEDAL_ASSERT(count > 0);

    array<byte_t> tmp_send_buffer;

    const std::int64_t rank = ctx_.get_this_thread_rank();
    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t size = dal::detail::check_mul_overflow(dtype_size, count);

    if (data_blocks_has_intersection(send_buf, recv_buf, size)) {
        // If send buffer is the same as recv buffer, the allreduce algorithm runs into
        // data races problem, so will need to create temporal copy of the send buffer
        tmp_send_buffer = array<byte_t>::empty(size);
        dal::detail::memcpy(dal::detail::default_host_policy{},
                            tmp_send_buffer.get_mutable_data(),
                            send_buf,
                            size);
        send_buffers_[rank] = buffer_info{ tmp_send_buffer.get_data(), count };
    }
    else {
        send_buffers_[rank] = buffer_info{ send_buf, count };
    }

    barrier_();

#ifdef ONEDAL_ENABLE_ASSERT
    if (rank == ctx_.get_root_rank()) {
        for (const auto& info : send_buffers_) {
            ONEDAL_ASSERT(info.count == count);
            ONEDAL_ASSERT(info.send_buf != nullptr);
        }
    }
#endif

    fill_with_zeros(recv_buf, count, dtype);
    for (const auto& info : send_buffers_) {
        ONEDAL_ASSERT(info.send_buf != recv_buf);
        reduce(info.send_buf, recv_buf, count, dtype, op);
    }

    barrier_([&]() {
        send_buffers_.clear();
        send_buffers_.resize(ctx_.get_thread_count());
    });
}

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
                                                std::int64_t count,
                                                const dal::detail::spmd_reduce_op& op_id) {
    ONEDAL_ASSERT(src_bytes);
    ONEDAL_ASSERT(dst_bytes);
    ONEDAL_ASSERT(count >= 0);

    const T* src = reinterpret_cast<const T*>(src_bytes);
    T* dst = reinterpret_cast<T*>(dst_bytes);

    switch_by_reduce_op<T>(op_id, [&](auto reduce_op) {
        reduce_op(src, dst, count);
    });
}

template <typename T>
void thread_communicator_allreduce::fill_with_zeros_impl(byte_t* dst_bytes, std::int64_t count) {
    ONEDAL_ASSERT(dst_bytes);
    ONEDAL_ASSERT(count >= 0);

    T* dst = reinterpret_cast<T*>(dst_bytes);
    for (std::int64_t i = 0; i < count; i++) {
        dst[i] = T(0);
    }
}

void thread_communicator_impl::barrier() {
    barrier_();
}

auto thread_communicator_impl::bcast(byte_t* send_buf,
                                     std::int64_t count,
                                     const data_type& dtype,
                                     std::int64_t root) -> request_t* {
    collective_operation_guard guard{ ctx_ };
    bcast_(send_buf, count, dtype, root);
    return nullptr;
}

#ifdef ONEDAL_DATA_PARALLEL
auto thread_communicator_impl::bcast(sycl::queue& q,
                                     byte_t* send_buf,
                                     std::int64_t count,
                                     const data_type& dtype,
                                     std::int64_t root) -> request_t* {
    ONEDAL_ASSERT(root >= 0);

    if (count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(count > 0);

    check_if_pointer_matches_queue(q, send_buf);

    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t size = dal::detail::check_mul_overflow(dtype_size, count);

    const auto send_buff_host = array<byte_t>::empty(size);
    if (get_rank() == root) {
        dal::detail::memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, size);
    }

    bcast(send_buff_host.get_mutable_data(), count, dtype, root);

    if (get_rank() != root) {
        dal::detail::memcpy_host2usm(q, send_buf, send_buff_host.get_mutable_data(), size);
    }
    return nullptr;
}
#endif

auto thread_communicator_impl::gather(const byte_t* send_buf,
                                      std::int64_t send_count,
                                      byte_t* recv_buf,
                                      std::int64_t recv_count,
                                      const data_type& dtype,
                                      std::int64_t root) -> request_t* {
    collective_operation_guard guard{ ctx_ };
    gather_(send_buf, send_count, recv_buf, recv_count, dtype, root);
    return nullptr;
}

#ifdef ONEDAL_DATA_PARALLEL
auto thread_communicator_impl::gather(sycl::queue& q,
                                      const byte_t* send_buf,
                                      std::int64_t send_count,
                                      byte_t* recv_buf,
                                      std::int64_t recv_count,
                                      const data_type& dtype,
                                      std::int64_t root) -> request_t* {
    ONEDAL_ASSERT(root >= 0);

    if (send_count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(send_count > 0);

    if (get_rank() == root) {
        ONEDAL_ASSERT(recv_buf);
        ONEDAL_ASSERT(recv_count > 0);
    }

    check_if_pointer_matches_queue(q, send_buf);
    check_if_pointer_matches_queue(q, recv_buf);

    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t send_size = dal::detail::check_mul_overflow(dtype_size, send_count);
    const std::int64_t recv_size = dal::detail::check_mul_overflow(dtype_size, recv_count);
    const std::int64_t all_recv_size = dal::detail::check_mul_overflow(get_rank_count(), recv_size);

    const auto send_buff_host = array<byte_t>::empty(send_size);
    dal::detail::memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, send_size);

    array<byte_t> recv_host;
    byte_t* recv_host_ptr = nullptr;
    if (get_rank() == root) {
        recv_host.reset(all_recv_size);
        recv_host_ptr = recv_host.get_mutable_data();
    }

    gather(send_buff_host.get_data(), send_count, recv_host_ptr, recv_count, dtype, root);

    if (get_rank() == root) {
        dal::detail::memcpy_host2usm(q, recv_buf, recv_host_ptr, all_recv_size);
    }

    return nullptr;
}
#endif

auto thread_communicator_impl::gatherv(const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       const std::int64_t* recv_counts,
                                       const std::int64_t* displs,
                                       const data_type& dtype,
                                       std::int64_t root) -> request_t* {
    collective_operation_guard guard{ ctx_ };
    gatherv_(send_buf, send_count, recv_buf, recv_counts, displs, dtype, root);
    return nullptr;
}

#ifdef ONEDAL_DATA_PARALLEL
auto thread_communicator_impl::gatherv(sycl::queue& q,
                                       const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       const std::int64_t* recv_counts_host,
                                       const std::int64_t* displs_host,
                                       const data_type& dtype,
                                       std::int64_t root) -> request_t* {
    ONEDAL_ASSERT(root >= 0);

    if (send_count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(send_count > 0);

    if (get_rank() == root) {
        ONEDAL_ASSERT(recv_buf);
        ONEDAL_ASSERT(recv_counts_host);
        ONEDAL_ASSERT(displs_host);
    }

    check_if_pointer_matches_queue(q, send_buf);
    check_if_pointer_matches_queue(q, recv_buf);

    const std::int64_t rank_count = get_rank_count();
    std::int64_t total_recv_count = 0;

    array<std::int64_t> displs_host_0;
    if (get_rank() == root) {
        displs_host_0.reset(rank_count);
        std::int64_t* displs_host_0_ptr = displs_host_0.get_mutable_data();

        for (std::int64_t i = 0; i < rank_count; i++) {
            displs_host_0_ptr[i] = total_recv_count;
            total_recv_count += recv_counts_host[i];
        }
    }

    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t send_size = dal::detail::check_mul_overflow(dtype_size, send_count);
    const std::int64_t total_recv_size =
        dal::detail::check_mul_overflow(dtype_size, total_recv_count);

    const auto send_buff_host = array<byte_t>::empty(send_size);
    dal::detail::memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, send_size);

    array<byte_t> recv_buf_host;
    byte_t* recv_buf_host_ptr = nullptr;
    if (get_rank() == root) {
        ONEDAL_ASSERT(total_recv_size > 0);
        recv_buf_host.reset(total_recv_size);
        recv_buf_host_ptr = recv_buf_host.get_mutable_data();
    }

    gatherv(send_buff_host.get_data(),
            send_count,
            recv_buf_host_ptr,
            recv_counts_host,
            displs_host_0.get_data(),
            dtype,
            root);

    if (get_rank() == root) {
        const std::int64_t* displs_host_0_ptr = displs_host_0.get_data();
        ONEDAL_ASSERT(displs_host_0_ptr);
        ONEDAL_ASSERT(displs_host);
        ONEDAL_ASSERT(recv_counts_host);

        for (std::int64_t i = 0; i < rank_count; i++) {
            const std::int64_t src_offset =
                dal::detail::check_mul_overflow(dtype_size, displs_host_0_ptr[i]);
            const std::int64_t dst_offset =
                dal::detail::check_mul_overflow(dtype_size, displs_host[i]);
            const std::int64_t copy_size =
                dal::detail::check_mul_overflow(dtype_size, recv_counts_host[i]);

            dal::detail::memcpy_host2usm(q,
                                         recv_buf + dst_offset,
                                         recv_buf_host_ptr + src_offset,
                                         copy_size);
        }
    }

    return nullptr;
}
#endif

auto thread_communicator_impl::allreduce(const byte_t* send_buf,
                                         byte_t* recv_buf,
                                         std::int64_t count,
                                         const data_type& dtype,
                                         const dal::detail::spmd_reduce_op& op) -> request_t* {
    collective_operation_guard guard{ ctx_ };
    allreduce_(send_buf, recv_buf, count, dtype, op);
    return nullptr;
}

#ifdef ONEDAL_DATA_PARALLEL
auto thread_communicator_impl::allreduce(sycl::queue& q,
                                         const byte_t* send_buf,
                                         byte_t* recv_buf,
                                         std::int64_t count,
                                         const data_type& dtype,
                                         const dal::detail::spmd_reduce_op& op) -> request_t* {
    if (count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(recv_buf);
    ONEDAL_ASSERT(count > 0);

    check_if_pointer_matches_queue(q, send_buf);
    check_if_pointer_matches_queue(q, recv_buf);

    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t byte_count = dal::detail::check_mul_overflow(dtype_size, count);

    const auto send_buff_host = array<byte_t>::empty(byte_count);
    const auto recv_buf_host = array<byte_t>::empty(byte_count);

    dal::detail::memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, byte_count);
    allreduce(send_buff_host.get_data(), recv_buf_host.get_mutable_data(), count, dtype, op);
    dal::detail::memcpy_host2usm(q, recv_buf, recv_buf_host.get_data(), byte_count);

    return nullptr;
}
#endif

auto thread_communicator_impl::allgather(const byte_t* send_buf,
                                         std::int64_t send_count,
                                         byte_t* recv_buf,
                                         std::int64_t recv_count,
                                         const data_type& dtype) -> request_t* {
    collective_operation_guard guard{ ctx_ };
    allgather_(send_buf, send_count, recv_buf, recv_count, dtype);
    return nullptr;
}

#ifdef ONEDAL_DATA_PARALLEL
auto thread_communicator_impl::allgather(sycl::queue& q,
                                         const byte_t* send_buf,
                                         std::int64_t send_count,
                                         byte_t* recv_buf,
                                         std::int64_t recv_count,
                                         const data_type& dtype) -> request_t* {
    if (send_count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(send_count > 0);
    ONEDAL_ASSERT(recv_buf);
    ONEDAL_ASSERT(recv_count > 0);

    check_if_pointer_matches_queue(q, send_buf);
    check_if_pointer_matches_queue(q, recv_buf);

    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t send_size = dal::detail::check_mul_overflow(dtype_size, send_count);
    const std::int64_t recv_size = dal::detail::check_mul_overflow(dtype_size, recv_count);
    const std::int64_t all_recv_size = dal::detail::check_mul_overflow(recv_size, get_rank_count());

    const auto send_buff_host = array<byte_t>::empty(send_size);
    const auto recv_buf_host = array<byte_t>::empty(all_recv_size);

    dal::detail::memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, send_size);
    allgather(send_buff_host.get_data(),
              send_count,
              recv_buf_host.get_mutable_data(),
              recv_count,
              dtype);
    dal::detail::memcpy_host2usm(q, recv_buf, recv_buf_host.get_data(), all_recv_size);

    return nullptr;
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
void thread_communicator_impl::check_if_pointer_matches_queue(const sycl::queue& q,
                                                              const void* pointer) {
    const auto pointer_type = sycl::get_pointer_type(pointer, q.get_context());
    if (pointer && pointer_type == sycl::usm::alloc::unknown) {
        throw std::runtime_error{ "Pointer passed to thread communicator "
                                  "is inaccessible in the given DPC++ context" };
    }
}
#endif

} // namespace oneapi::dal::test::engine
