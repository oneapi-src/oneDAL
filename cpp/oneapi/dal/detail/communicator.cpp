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

#include "oneapi/dal/detail/communicator.hpp"

namespace oneapi::dal::detail::v1 {

#ifdef ONEDAL_DATA_PARALLEL
static void check_if_pointer_matches_queue(const sycl::queue& q, const void* ptr) {
    if (ptr) {
        const auto alloc_kind = sycl::get_pointer_type(ptr, q.get_context());
        if (alloc_kind == sycl::usm::alloc::unknown) {
            throw invalid_argument{ error_messages::unknown_usm_pointer_type() };
        }
    }
}
#endif

static void wait_request(spmd_request_iface* request) {
    if (request) {
        request->wait();
    }
}

const char* communication_error::what() const noexcept {
    return std::runtime_error::what();
}

#ifdef ONEDAL_DATA_PARALLEL
auto spmd_communicator_via_host_impl::bcast(sycl::queue& q,
                                            byte_t* send_buf,
                                            std::int64_t count,
                                            const data_type& dtype,
                                            const std::vector<sycl::event>& deps,
                                            std::int64_t root) -> spmd_request_iface* {
    ONEDAL_ASSERT(root >= 0);

    if (count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(count > 0);

    check_if_pointer_matches_queue(q, send_buf);
    sycl::event::wait_and_throw(deps);

    const std::int64_t dtype_size = get_data_type_size(dtype);
    const std::int64_t size = check_mul_overflow(dtype_size, count);

    const auto send_buff_host = array<byte_t>::empty(size);
    if (get_rank() == root) {
        memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, size);
    }

    wait_request(bcast(send_buff_host.get_mutable_data(), count, dtype, root));

    if (get_rank() != root) {
        memcpy_host2usm(q, send_buf, send_buff_host.get_mutable_data(), size);
    }

    return nullptr;
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
auto spmd_communicator_via_host_impl::gather(sycl::queue& q,
                                             const byte_t* send_buf,
                                             std::int64_t send_count,
                                             byte_t* recv_buf,
                                             std::int64_t recv_count,
                                             const data_type& dtype,
                                             const std::vector<sycl::event>& deps,
                                             std::int64_t root) -> spmd_request_iface* {
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
    sycl::event::wait_and_throw(deps);

    const std::int64_t dtype_size = get_data_type_size(dtype);
    const std::int64_t send_size = check_mul_overflow(dtype_size, send_count);
    const std::int64_t recv_size = check_mul_overflow(dtype_size, recv_count);
    const std::int64_t all_recv_size = check_mul_overflow(get_rank_count(), recv_size);

    const auto send_buff_host = array<byte_t>::empty(send_size);
    memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, send_size);

    array<byte_t> recv_host;
    byte_t* recv_host_ptr = nullptr;
    if (get_rank() == root) {
        recv_host.reset(all_recv_size);
        recv_host_ptr = recv_host.get_mutable_data();
    }

    wait_request(
        gather(send_buff_host.get_data(), send_count, recv_host_ptr, recv_count, dtype, root));

    if (get_rank() == root) {
        memcpy_host2usm(q, recv_buf, recv_host_ptr, all_recv_size);
    }

    return nullptr;
}

#endif

#ifdef ONEDAL_DATA_PARALLEL
auto spmd_communicator_via_host_impl::gatherv(sycl::queue& q,
                                              const byte_t* send_buf,
                                              std::int64_t send_count,
                                              byte_t* recv_buf,
                                              const std::int64_t* recv_counts_host,
                                              const std::int64_t* displs_host,
                                              const data_type& dtype,
                                              const std::vector<sycl::event>& deps,
                                              std::int64_t root) -> spmd_request_iface* {
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
    sycl::event::wait_and_throw(deps);

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

    const std::int64_t dtype_size = get_data_type_size(dtype);
    const std::int64_t send_size = check_mul_overflow(dtype_size, send_count);
    const std::int64_t total_recv_size = check_mul_overflow(dtype_size, total_recv_count);

    const auto send_buff_host = array<byte_t>::empty(send_size);
    memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, send_size);

    array<byte_t> recv_buf_host;
    byte_t* recv_buf_host_ptr = nullptr;
    if (get_rank() == root) {
        ONEDAL_ASSERT(total_recv_size > 0);
        recv_buf_host.reset(total_recv_size);
        recv_buf_host_ptr = recv_buf_host.get_mutable_data();
    }

    wait_request(gatherv(send_buff_host.get_data(),
                         send_count,
                         recv_buf_host_ptr,
                         recv_counts_host,
                         displs_host_0.get_data(),
                         dtype,
                         root));

    if (get_rank() == root) {
        const std::int64_t* displs_host_0_ptr = displs_host_0.get_data();
        ONEDAL_ASSERT(displs_host_0_ptr);
        ONEDAL_ASSERT(displs_host);
        ONEDAL_ASSERT(recv_counts_host);

        for (std::int64_t i = 0; i < rank_count; i++) {
            const std::int64_t src_offset = check_mul_overflow(dtype_size, displs_host_0_ptr[i]);
            const std::int64_t dst_offset = check_mul_overflow(dtype_size, displs_host[i]);
            const std::int64_t copy_size = check_mul_overflow(dtype_size, recv_counts_host[i]);

            memcpy_host2usm(q, recv_buf + dst_offset, recv_buf_host_ptr + src_offset, copy_size);
        }
    }

    return nullptr;
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
auto spmd_communicator_via_host_impl::allgather(sycl::queue& q,
                                                const byte_t* send_buf,
                                                std::int64_t send_count,
                                                byte_t* recv_buf,
                                                std::int64_t recv_count,
                                                const data_type& dtype,
                                                const std::vector<sycl::event>& deps)
    -> spmd_request_iface* {
    if (send_count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(send_count > 0);
    ONEDAL_ASSERT(recv_buf);
    ONEDAL_ASSERT(recv_count > 0);

    check_if_pointer_matches_queue(q, send_buf);
    check_if_pointer_matches_queue(q, recv_buf);
    sycl::event::wait_and_throw(deps);

    const std::int64_t dtype_size = get_data_type_size(dtype);
    const std::int64_t send_size = check_mul_overflow(dtype_size, send_count);
    const std::int64_t recv_size = check_mul_overflow(dtype_size, recv_count);
    const std::int64_t all_recv_size = check_mul_overflow(recv_size, get_rank_count());

    const auto send_buff_host = array<byte_t>::empty(send_size);
    const auto recv_buf_host = array<byte_t>::empty(all_recv_size);

    memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, send_size);
    wait_request(allgather(send_buff_host.get_data(),
                           send_count,
                           recv_buf_host.get_mutable_data(),
                           recv_count,
                           dtype));
    memcpy_host2usm(q, recv_buf, recv_buf_host.get_data(), all_recv_size);

    return nullptr;
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
auto spmd_communicator_via_host_impl::allreduce(sycl::queue& q,
                                                const byte_t* send_buf,
                                                byte_t* recv_buf,
                                                std::int64_t count,
                                                const data_type& dtype,
                                                const spmd_reduce_op& op,
                                                const std::vector<sycl::event>& deps)
    -> spmd_request_iface* {
    if (count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(recv_buf);
    ONEDAL_ASSERT(count > 0);

    check_if_pointer_matches_queue(q, send_buf);
    check_if_pointer_matches_queue(q, recv_buf);
    sycl::event::wait_and_throw(deps);

    const std::int64_t dtype_size = get_data_type_size(dtype);
    const std::int64_t byte_count = check_mul_overflow(dtype_size, count);

    const auto send_buff_host = array<byte_t>::empty(byte_count);
    const auto recv_buf_host = array<byte_t>::empty(byte_count);

    memcpy_usm2host(q, send_buff_host.get_mutable_data(), send_buf, byte_count);
    wait_request(
        allreduce(send_buff_host.get_data(), recv_buf_host.get_mutable_data(), count, dtype, op));
    memcpy_host2usm(q, recv_buf, recv_buf_host.get_data(), byte_count);

    return nullptr;
}
#endif

spmd_request::spmd_request() : impl_(nullptr) {}

void spmd_request::wait() {
    if (impl_) {
        return impl_->wait();
    }
}

bool spmd_request::test() {
    if (impl_) {
        return impl_->test();
    }
    return true;
}

std::int64_t spmd_communicator::get_rank() const {
    return impl_->get_rank();
}

std::int64_t spmd_communicator::get_rank_count() const {
    return impl_->get_rank_count();
}

std::int64_t spmd_communicator::get_default_root_rank() const {
    return impl_->get_default_root_rank();
}

void spmd_communicator::barrier() const {
    return impl_->barrier();
}

spmd_request spmd_communicator::bcast(byte_t* send_buf,
                                      std::int64_t count,
                                      const data_type& dtype,
                                      std::int64_t root) const {
    return make_private<spmd_request>(impl_->bcast(send_buf, count, dtype, fix_root_rank(root)));
}

#ifdef ONEDAL_DATA_PARALLEL
spmd_request spmd_communicator::bcast(sycl::queue& q,
                                      byte_t* send_buf,
                                      std::int64_t count,
                                      const data_type& dtype,
                                      const std::vector<sycl::event>& deps,
                                      std::int64_t root) const {
    return make_private<spmd_request>(
        impl_->bcast(q, send_buf, count, dtype, deps, fix_root_rank(root)));
}
#endif

spmd_request spmd_communicator::gather(const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       std::int64_t recv_count,
                                       const data_type& dtype,
                                       std::int64_t root) const {
    return make_private<spmd_request>(
        impl_->gather(send_buf, send_count, recv_buf, recv_count, dtype, fix_root_rank(root)));
}

#ifdef ONEDAL_DATA_PARALLEL
spmd_request spmd_communicator::gather(sycl::queue& q,
                                       const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       std::int64_t recv_count,
                                       const data_type& dtype,
                                       const std::vector<sycl::event>& deps,
                                       std::int64_t root) const {
    return make_private<spmd_request>(impl_->gather(q,
                                                    send_buf,
                                                    send_count,
                                                    recv_buf,
                                                    recv_count,
                                                    dtype,
                                                    deps,
                                                    fix_root_rank(root)));
}
#endif

spmd_request spmd_communicator::gatherv(const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_counts,
                                        const std::int64_t* displs,
                                        const data_type& dtype,
                                        std::int64_t root) const {
    return make_private<spmd_request>(impl_->gatherv(send_buf,
                                                     send_count,
                                                     recv_buf,
                                                     recv_counts,
                                                     displs,
                                                     dtype,
                                                     fix_root_rank(root)));
}

#ifdef ONEDAL_DATA_PARALLEL
spmd_request spmd_communicator::gatherv(sycl::queue& q,
                                        const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_counts,
                                        const std::int64_t* displs,
                                        const data_type& dtype,
                                        const std::vector<sycl::event>& deps,
                                        std::int64_t root) const {
    return make_private<spmd_request>(impl_->gatherv(q,
                                                     send_buf,
                                                     send_count,
                                                     recv_buf,
                                                     recv_counts,
                                                     displs,
                                                     dtype,
                                                     deps,
                                                     fix_root_rank(root)));
}
#endif

spmd_request spmd_communicator::allreduce(const byte_t* send_buf,
                                          byte_t* recv_buf,
                                          std::int64_t count,
                                          const data_type& dtype,
                                          const spmd_reduce_op& op) const {
    return make_private<spmd_request>(impl_->allreduce(send_buf, recv_buf, count, dtype, op));
}

#ifdef ONEDAL_DATA_PARALLEL
spmd_request spmd_communicator::allreduce(sycl::queue& q,
                                          const byte_t* send_buf,
                                          byte_t* recv_buf,
                                          std::int64_t count,
                                          const data_type& dtype,
                                          const spmd_reduce_op& op,
                                          const std::vector<sycl::event>& deps) const {
    return make_private<spmd_request>(
        impl_->allreduce(q, send_buf, recv_buf, count, dtype, op, deps));
}
#endif

spmd_request spmd_communicator::allgather(const byte_t* send_buf,
                                          std::int64_t send_count,
                                          byte_t* recv_buf,
                                          std::int64_t recv_count,
                                          const data_type& dtype) const {
    return make_private<spmd_request>(
        impl_->allgather(send_buf, send_count, recv_buf, recv_count, dtype));
}

#ifdef ONEDAL_DATA_PARALLEL
spmd_request spmd_communicator::allgather(sycl::queue& q,
                                          const byte_t* send_buf,
                                          std::int64_t send_count,
                                          byte_t* recv_buf,
                                          std::int64_t recv_count,
                                          const data_type& dtype,
                                          const std::vector<sycl::event>& deps) const {
    return make_private<spmd_request>(
        impl_->allgather(q, send_buf, send_count, recv_buf, recv_count, dtype, deps));
}
#endif

} // namespace oneapi::dal::detail::v1
