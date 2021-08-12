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

#pragma once

#include <mpi.h>
#include "oneapi/dal/detail/communicator.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class mpi_error : public communication_error {};

inline void mpi_call(int mpi_status) {}

inline MPI_Datatype mpi_data_type(const data_type& dtype) {
    switch (dtype) {
        case data_type::int8: return MPI_INT8_T;
        case data_type::int16: return MPI_INT16_T;
        case data_type::int32: return MPI_INT32_T;
        case data_type::int64: return MPI_INT64_T;
        case data_type::uint8: return MPI_UINT8_T;
        case data_type::uint16: return MPI_UINT16_T;
        case data_type::uint32: return MPI_UINT32_T;
        case data_type::uint64: return MPI_UINT64_T;
        case data_type::float32: return MPI_FLOAT;
        case data_type::float64: return MPI_DOUBLE;
        case data_type::bfloat16: return MPI_UINT16_T;
        default: ONEDAL_ASSERT(!"Unknown data type");
    }
    return MPI_DATATYPE_NULL;
}

#ifdef ONEDAL_DATA_PARALLEL
class host2usm_synchronizing_buffer {
public:
    explicit host2usm_synchronizing_buffer(const sycl::queue& q,
                                           byte_t* usm_ptr,
                                           const data_type& dtype,
                                           std::int64_t count)
            : queue_(q),
              usm_ptr_(usm_ptr),
              host_ptr_(nullptr),
              dtype_(dtype),
              count_(count),
              is_readonly_(false) {
        ONEDAL_ASSERT(usm_ptr);
        ONEDAL_ASSERT(count > 0);
    }

    explicit host2usm_synchronizing_buffer(const sycl::queue& q,
                                           const byte_t* usm_ptr,
                                           const data_type& dtype,
                                           std::int64_t count)
            : queue_(q),
              usm_ptr_(const_cast<byte_t*>(usm_ptr)),
              host_ptr_(nullptr),
              dtype_(dtype),
              count_(count),
              is_readonly_(true) {
        ONEDAL_ASSERT(usm_ptr);
        ONEDAL_ASSERT(count > 0);
    }

    host2usm_synchronizing_buffer(const host2usm_synchronizing_buffer&) = delete;
    host2usm_synchronizing_buffer& operator=(const host2usm_synchronizing_buffer&) = delete;

    ~host2usm_synchronizing_buffer() {
        // If destructor called because of exception,
        // we don't need to write back anything
        if (!std::uncaught_exception()) {
            try_write_back();
        }

        free(default_host_policy{}, host_ptr_);
        host_ptr_ = nullptr;
    }

    const byte_t* get_host_data() {
        ONEDAL_ASSERT(!is_readonly_);
        return try_copy_to_host();
    }

    byte_t* get_mutable_host_data() {
        ONEDAL_ASSERT(!is_readonly_);
        need_write_back_ = true;
        return try_copy_to_host();
    }

    void write_back() {
        try_write_back();
    }

private:
    byte_t* try_copy_to_host() {
        if (!host_ptr_) {
            const std::int64_t dtype_size = get_data_type_size(dtype_);
            const std::int64_t size = check_mul_overflow(dtype_size, count_);

            host_ptr_ = static_cast<byte_t*>(malloc(default_host_policy{}, size));
            memcpy_usm2host(queue_, host_ptr_, usm_ptr_, size);
        }
        return host_ptr_;
    }

    void try_write_back() {
        // Write-back happens only if mutable host data was requested
        if (host_ptr_ && need_write_back_) {
            const std::int64_t dtype_size = get_data_type_size(dtype_);
            const std::int64_t size = check_mul_overflow(dtype_size, count_);
            memcpy_host2usm(queue_, usm_ptr_, host_ptr_, size);
            need_write_back_ = false;
        }
    }

    sycl::queue queue_;
    byte_t* usm_ptr_;
    byte_t* host_ptr_;
    data_type dtype_;
    std::int64_t count_;
    bool is_readonly_;
    bool need_write_back_ = false;
};
#endif

class mpi_request_impl : public dal::detail::spmd_request_iface {
public:
    explicit mpi_request_impl(const MPI_Request& request) : mpi_request_(request) {}

    mpi_request_impl(const mpi_request_impl&) = delete;
    mpi_request_impl& operator=(const mpi_request_impl&) = delete;

    void wait() override {
        mpi_call(MPI_Wait(&mpi_request_, MPI_STATUS_IGNORE));
    }

    bool test() override {
        int flag;
        mpi_call(MPI_Test(&mpi_request_, &flag, MPI_STATUS_IGNORE));
        return bool(flag != 0);
    }

private:
    MPI_Request mpi_request_;
};

class mpi_communicator_impl : public spmd_communicator_iface {
public:
    explicit mpi_communicator_impl(const MPI_Comm& mpi_comm, std::int64_t default_root = 0)
            : mpi_comm_(mpi_comm),
              default_root_(default_root) {}

    std::int64_t get_rank() override {
        if (rank_ < 0) {
            int rank;
            mpi_call(MPI_Comm_rank(mpi_comm_, &rank));
            rank_ = rank;
        }
        return rank_;
    }

    std::int64_t get_rank_count() override {
        if (rank_count_ < 0) {
            int rank_count;
            mpi_call(MPI_Comm_size(mpi_comm_, &rank_count));
            rank_count_ = rank_count;
        }
        return rank_count_;
    }

    std::int64_t get_default_root_rank() override {
        return default_root_;
    }

    void barrier() override {
        mpi_call(MPI_Barrier(mpi_comm_));
    }

    spmd_request_iface* bcast(byte_t* send_buf,
                              std::int64_t count,
                              const data_type& dtype,
                              std::int64_t root) override {
        ONEDAL_ASSERT(root >= 0);

        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(count > 0);

        MPI_Request mpi_request;
        mpi_call(MPI_Ibcast(send_buf,
                            integral_cast<int>(count),
                            mpi_data_type(dtype),
                            integral_cast<int>(root),
                            mpi_comm_,
                            &mpi_request));

        return new mpi_request_impl{ mpi_request };
    }

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request_iface* bcast(sycl::queue& q,
                              byte_t* send_buf,
                              std::int64_t count,
                              const data_type& dtype,
                              const std::vector<sycl::event>& deps,
                              std::int64_t root) override {
        ONEDAL_ASSERT(root >= 0);

        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(count > 0);

        check_if_queue_matches_pointer(q, send_buf);
        sycl::event::wait_and_throw(deps);

        host2usm_synchronizing_buffer sync_send_buf(q, send_buf, dtype, count);
        wait_request(bcast(sync_send_buf.get_mutable_host_data(), count, dtype, root));
    }
#endif

    spmd_request_iface* gather(const byte_t* send_buf,
                               std::int64_t send_count,
                               byte_t* recv_buf,
                               std::int64_t recv_count,
                               const data_type& dtype,
                               std::int64_t root) override {}

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request_iface* gather(sycl::queue& q,
                               const byte_t* send_buf,
                               std::int64_t send_count,
                               byte_t* recv_buf,
                               std::int64_t recv_count,
                               const data_type& dtype,
                               const std::vector<sycl::event>& deps,
                               std::int64_t root) override {}

#endif

    spmd_request_iface* gatherv(const byte_t* send_buf,
                                std::int64_t send_count,
                                byte_t* recv_buf,
                                const std::int64_t* recv_counts,
                                const std::int64_t* displs,
                                const data_type& dtype,
                                std::int64_t root) override {}

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request_iface* gatherv(sycl::queue& q,
                                const byte_t* send_buf,
                                std::int64_t send_count,
                                byte_t* recv_buf,
                                const std::int64_t* recv_counts_host,
                                const std::int64_t* displs_host,
                                const data_type& dtype,
                                const std::vector<sycl::event>& deps,
                                std::int64_t root) override {}
#endif

    spmd_request_iface* allgather(const byte_t* send_buf,
                                  std::int64_t send_count,
                                  byte_t* recv_buf,
                                  std::int64_t recv_count,
                                  const data_type& dtype) override {}

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request_iface* allgather(sycl::queue& q,
                                  const byte_t* send_buf,
                                  std::int64_t send_count,
                                  byte_t* recv_buf,
                                  std::int64_t recv_count,
                                  const data_type& dtype,
                                  const std::vector<sycl::event>& deps) override {}
#endif

    spmd_request_iface* allreduce(const byte_t* send_buf,
                                  byte_t* recv_buf,
                                  std::int64_t count,
                                  const data_type& dtype,
                                  const spmd_reduce_op& op) override {}

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request_iface* allreduce(sycl::queue& q,
                                  const byte_t* send_buf,
                                  byte_t* recv_buf,
                                  std::int64_t count,
                                  const data_type& dtype,
                                  const spmd_reduce_op& op,
                                  const std::vector<sycl::event>& deps) override {}
#endif

private:
#ifdef ONEDAL_DATA_PARALLEL
    void check_if_pointer_matches_queue(const sycl::queue& q, void* ptr) {
        const auto alloc_kind = sycl::get_pointer_type(ptr, q.get_context());
        if (alloc_kind == sycl::usm::alloc::unknown) {
            throw invalid_argument{ error_messages::unknown_usm_pointer_type() };
        }
    }
#endif

    MPI_Comm mpi_comm_;
    std::int64_t default_root_;
    std::int64_t rank_ = -1;
    std::int64_t rank_count_ = -1;
};

class mpi_communicator : public spmd_communicator {
public:
    explicit mpi_communicator(const MPI_Comm& mpi_comm)
            : spmd_communicator(new mpi_communicator_impl{ mp_comm }) {}
};

} // namespace v1
} // namespace oneapi::dal::detail
