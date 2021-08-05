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

#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend {

class fake_spmd_communicator_impl : public dal::detail::spmd_communicator_iface {
public:
    using base_t = dal::detail::spmd_communicator_iface;
    using request_t = dal::detail::spmd_request_iface;

    static constexpr std::int64_t root_rank = 0;
    static constexpr std::int64_t rank_count = 1;

    std::int64_t get_rank() override {
        return root_rank;
    }

    std::int64_t get_default_root_rank() override {
        return root_rank;
    }

    std::int64_t get_rank_count() override {
        return rank_count;
    }

    void barrier() override {}

    request_t* bcast(byte_t* send_buf,
                     std::int64_t count,
                     const data_type& dtype,
                     std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* bcast(sycl::queue& q,
                     byte_t* send_buf,
                     std::int64_t count,
                     const data_type& dtype,
                     std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        return nullptr;
    }
#endif

    request_t* gather(const byte_t* send_buf,
                      std::int64_t send_count,
                      byte_t* recv_buf,
                      std::int64_t recv_count,
                      const data_type& dtype,
                      std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        ONEDAL_ASSERT(send_count == recv_count);

        copy_if_different_pointers(recv_buf, send_buf, send_count, dtype);

        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* gather(sycl::queue& q,
                      const byte_t* send_buf,
                      std::int64_t send_count,
                      byte_t* recv_buf,
                      std::int64_t recv_count,
                      const data_type& dtype,
                      std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        ONEDAL_ASSERT(send_count == recv_count);

        copy_if_different_pointers(q, recv_buf, send_buf, send_count, dtype);

        return nullptr;
    }

#endif

    request_t* gatherv(const byte_t* send_buf,
                       std::int64_t send_count,
                       byte_t* recv_buf,
                       const std::int64_t* recv_counts,
                       const std::int64_t* displs,
                       const data_type& dtype,
                       std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);

        if (send_count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(recv_counts);
        ONEDAL_ASSERT(displs);
        ONEDAL_ASSERT(recv_counts[0] == send_count);

        copy_if_different_pointers(recv_buf + displs[0], send_buf, send_count, dtype);

        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* gatherv(sycl::queue& q,
                       const byte_t* send_buf,
                       std::int64_t send_count,
                       byte_t* recv_buf,
                       const std::int64_t* recv_counts,
                       const std::int64_t* displs,
                       const data_type& dtype,
                       std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);

        if (send_count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(recv_counts);
        ONEDAL_ASSERT(displs);
        ONEDAL_ASSERT(recv_counts[0] == send_count);

        copy_if_different_pointers(q, recv_buf + displs[0], send_buf, send_count, dtype);

        return nullptr;
    }
#endif

    request_t* allgather(const byte_t* send_buf,
                         std::int64_t send_count,
                         byte_t* recv_buf,
                         std::int64_t recv_count,
                         const data_type& dtype) override {
        ONEDAL_ASSERT(send_count == recv_count);

        copy_if_different_pointers(recv_buf, send_buf, send_count, dtype);

        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* allgather(sycl::queue& q,
                         const byte_t* send_buf,
                         std::int64_t send_count,
                         byte_t* recv_buf,
                         std::int64_t recv_count,
                         const data_type& dtype) override {
        ONEDAL_ASSERT(send_count == recv_count);

        copy_if_different_pointers(q, recv_buf, send_buf, send_count, dtype);

        return nullptr;
    }
#endif

    request_t* allreduce(const byte_t* send_buf,
                         byte_t* recv_buf,
                         std::int64_t count,
                         const data_type& dtype,
                         const dal::detail::spmd_reduce_op& op) override {
        copy_if_different_pointers(recv_buf, send_buf, count, dtype);
        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* allreduce(sycl::queue& q,
                         const byte_t* send_buf,
                         byte_t* recv_buf,
                         std::int64_t count,
                         const data_type& dtype,
                         const dal::detail::spmd_reduce_op& op) override {
        copy_if_different_pointers(recv_buf, send_buf, count, dtype);
        return nullptr;
    }
#endif

    void copy_if_different_pointers(byte_t* dst,
                                    const byte_t* src,
                                    std::int64_t count,
                                    const data_type& dtype) const {
        if (count == 0) {
            return;
        }

        ONEDAL_ASSERT(src);
        ONEDAL_ASSERT(dst);
        ONEDAL_ASSERT(count > 0);

        if (dst == src) {
            return;
        }

        const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
        const std::int64_t size = dal::detail::check_mul_overflow(dtype_size, count);
        memcpy(dst, src, size);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void copy_if_different_pointers(sycl::queue& q,
                                    byte_t* dst,
                                    const byte_t* src,
                                    std::int64_t count,
                                    const data_type& dtype) const {
        if (count == 0) {
            return;
        }

        ONEDAL_ASSERT(src);
        ONEDAL_ASSERT(dst);
        ONEDAL_ASSERT(count > 0);
        ONEDAL_ASSERT(is_known_usm(q, src));
        ONEDAL_ASSERT(is_known_usm(q, dst));

        if (dst == src) {
            return;
        }

        const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
        const std::int64_t size = dal::detail::check_mul_overflow(dtype_size, count);
        memcpy(q, dst, src, size).wait_and_throw();
    }
#endif
};

fake_spmd_communicator::fake_spmd_communicator()
        : dal::detail::spmd_communicator(new fake_spmd_communicator_impl{}) {}

} // namespace oneapi::dal::backend
