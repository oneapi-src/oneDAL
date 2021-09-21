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

// IMPORTANT! This file should not be included to any other non-mpi header,
// otherwise it creates dependeny on MPI at the user's application compile time
// TODO: In the future this can be solved via __has_include C++17 feature

#include <mpi.h>
#include <ccl.hpp>
#include "oneapi/dal/detail/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::detail {
namespace v1 {

static void check_if_pointer_matches_queue(const sycl::queue& q, const void* ptr) {
    if (ptr) {
        const auto alloc_kind = sycl::get_pointer_type(ptr, q.get_context());
        if (alloc_kind == sycl::usm::alloc::unknown) {
            throw invalid_argument{ error_messages::unknown_usm_pointer_type() };
        }
    }
}

inline ccl::datatype make_oneccl_data_type(const data_type& dtype) {
    switch (dtype) {
        case data_type::int8: return ccl::datatype::int8;
        case data_type::int16: return ccl::datatype::int16;
        case data_type::int32: return ccl::datatype::int32;
        case data_type::int64: return ccl::datatype::int64;
        case data_type::uint8: return ccl::datatype::uint8;
        case data_type::uint16: return ccl::datatype::uint16;
        case data_type::uint32: return ccl::datatype::uint32;
        case data_type::uint64: return ccl::datatype::uint64;
        case data_type::float32: return ccl::datatype::float32;
        case data_type::float64: return ccl::datatype::float64;
        case data_type::bfloat16: return ccl::datatype::uint16;
        default: throw communication_error("Unknown data type");
    }
}

inline ccl::reduction make_oneccl_reduce_op(const spmd_reduce_op& op) {
    switch (op) {
        case spmd_reduce_op::sum: return ccl::reduction::sum;
        default: throw communication_error("Unknown reduce operation");
    }
}

class oneccl_request_impl : public dal::detail::spmd_request_iface {
public:
    explicit oneccl_request_impl(ccl::event&& request) {
        event_ = std::move(request);
    }

    oneccl_request_impl(const oneccl_request_impl&) = delete;
    oneccl_request_impl& operator=(const oneccl_request_impl&) = delete;

    void wait() override {
        event_.wait();
    }

    bool test() override {
        return event_.test();
    }

private:
    ccl::event event_;
};

class oneccl_comm_wrapper {
public:
    explicit oneccl_comm_wrapper(ccl::communicator&& comm) : comm_(std::move(comm)) {}
    ccl::communicator& get_ref() {
        return comm_;
    }

private:
    ccl::communicator comm_;
};

class oneccl_stream_wrapper {
public:
    explicit oneccl_stream_wrapper(ccl::stream&& stream) : stream_(std::move(stream)) {}
    ccl::stream& get_ref() {
        return stream_;
    }

private:
    ccl::stream stream_;
};

/// Implementation of the low-level SPMD communicator interface via oneCCL
/// TODO: Currently message sizes are limited via `int` type.
///       Large message sizes should be handled on the communicator side in the future.
class oneccl_communicator_impl : public spmd_communicator_iface {
public:
    // Explicitly declare all virtual functions with overloads to workaround Clang warning
    // https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
    using spmd_communicator_iface::bcast;
    using spmd_communicator_iface::gather;
    using spmd_communicator_iface::gatherv;
    using spmd_communicator_iface::allgather;
    using spmd_communicator_iface::allreduce;

    explicit oneccl_communicator_impl(const sycl::queue& queue, std::int64_t default_root = 0)
            : queue_(queue),
              default_root_(default_root) {
        init();
        auto dev = ccl::create_device(queue_.get_device());
        auto ctx = ccl::create_context(queue_.get_context());
        device_comm_.reset(new oneccl_comm_wrapper{
            ccl::create_communicator(rank_count_, rank_, dev, ctx, kvs_) });
        stream_.reset(new oneccl_stream_wrapper{ ccl::create_stream(queue_) });
        host_comm_.reset(
            new oneccl_comm_wrapper{ ccl::create_communicator(rank_count_, rank_, kvs_) });
    }

    std::int64_t get_rank() override {
        return rank_;
    }

    std::int64_t get_rank_count() override {
        return rank_count_;
    }

    std::int64_t get_default_root_rank() override {
        return default_root_;
    }

    void barrier() override {
        ccl::barrier(device_comm_->get_ref(), stream_->get_ref()).wait();
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

        auto event = ccl::broadcast(send_buf,
                                    integral_cast<int>(count),
                                    make_oneccl_data_type(dtype),
                                    integral_cast<int>(root),
                                    host_comm_->get_ref());
        return new oneccl_request_impl{ std::move(event) };
    }

    spmd_request_iface* bcast(sycl::queue& q,
                              byte_t* send_buf,
                              std::int64_t count,
                              const data_type& dtype,
                              const std::vector<sycl::event>& deps,
                              std::int64_t root) override {
        check_if_pointer_matches_queue(queue_, send_buf);
        check_if_pointer_matches_queue(q, send_buf);
        ONEDAL_ASSERT(root >= 0);

        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(count > 0);

        sycl::event::wait(deps);

        auto event = ccl::broadcast(send_buf,
                                    integral_cast<int>(count),
                                    make_oneccl_data_type(dtype),
                                    integral_cast<int>(root),
                                    device_comm_->get_ref(),
                                    stream_->get_ref());
        return new oneccl_request_impl{ std::move(event) };
    }

    spmd_request_iface* gather(const byte_t* send_buf,
                               std::int64_t send_count,
                               byte_t* recv_buf,
                               std::int64_t recv_count,
                               const data_type& dtype,
                               std::int64_t root) override {
        throw communication_error(
            "gatherX is not supported by oneCCL! Consider usage of allgatherX instead.");
    }

    spmd_request_iface* gather(sycl::queue& q,
                               const byte_t* send_buf,
                               std::int64_t send_count,
                               byte_t* recv_buf,
                               std::int64_t recv_count,
                               const data_type& dtype,
                               const std::vector<sycl::event>& deps,
                               std::int64_t root) override {
        throw communication_error(
            "gatherX is not supported by oneCCL! Consider usage of allgatherX instead.");
    }

    spmd_request_iface* gatherv(const byte_t* send_buf,
                                std::int64_t send_count,
                                byte_t* recv_buf,
                                const std::int64_t* recv_counts,
                                const std::int64_t* displs,
                                const data_type& dtype,
                                std::int64_t root) override {
        throw communication_error(
            "gatherX is not supported by oneCCL! Consider usage of allgatherX instead.");
    }

    spmd_request_iface* gatherv(sycl::queue& q,
                                const byte_t* send_buf,
                                std::int64_t send_count,
                                byte_t* recv_buf,
                                const std::int64_t* recv_counts_host,
                                const std::int64_t* displs_host,
                                const data_type& dtype,
                                const std::vector<sycl::event>& deps,
                                std::int64_t root) override {
        throw communication_error(
            "gatherX is not supported by oneCCL! Consider usage of allgatherX instead.");
    }

    spmd_request_iface* allgather(const byte_t* send_buf,
                                  std::int64_t send_count,
                                  byte_t* recv_buf,
                                  std::int64_t recv_count,
                                  const data_type& dtype) override {
        if (send_count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(recv_buf);

        std::vector<size_t> recv_counts(rank_count_, integral_cast<int>(recv_count));

        auto event = ccl::allgatherv(send_buf,
                                     integral_cast<int>(send_count),
                                     recv_buf,
                                     recv_counts,
                                     make_oneccl_data_type(dtype),
                                     host_comm_->get_ref());
        return new oneccl_request_impl{ std::move(event) };
    }

    spmd_request_iface* allgather(sycl::queue& q,
                                  const byte_t* send_buf,
                                  std::int64_t send_count,
                                  byte_t* recv_buf,
                                  std::int64_t recv_count,
                                  const data_type& dtype,
                                  const std::vector<sycl::event>& deps = {}) override {
        check_if_pointer_matches_queue(queue_, send_buf);
        check_if_pointer_matches_queue(queue_, recv_buf);
        if (send_count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(recv_buf);

        std::vector<size_t> recv_counts(rank_count_, integral_cast<int>(recv_count));

        auto event = ccl::allgatherv(send_buf,
                                     integral_cast<int>(send_count),
                                     recv_buf,
                                     recv_counts,
                                     make_oneccl_data_type(dtype),
                                     device_comm_->get_ref(),
                                     stream_->get_ref());
        return new oneccl_request_impl{ std::move(event) };
    }

    spmd_request_iface* allreduce(const byte_t* send_buf,
                                  byte_t* recv_buf,
                                  std::int64_t count,
                                  const data_type& dtype,
                                  const spmd_reduce_op& op) override {
        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(recv_buf);

        auto event = ccl::allreduce(send_buf,
                                    recv_buf,
                                    integral_cast<int>(count),
                                    make_oneccl_data_type(dtype),
                                    make_oneccl_reduce_op(op),
                                    host_comm_->get_ref());
        return new oneccl_request_impl{ std::move(event) };
    }

    /// `allreduce` that accepts USM pointers
    spmd_request_iface* allreduce(sycl::queue& q,
                                  const byte_t* send_buf,
                                  byte_t* recv_buf,
                                  std::int64_t count,
                                  const data_type& dtype,
                                  const spmd_reduce_op& op = spmd_reduce_op::sum,
                                  const std::vector<sycl::event>& deps = {}) override {
        check_if_pointer_matches_queue(queue_, send_buf);
        check_if_pointer_matches_queue(queue_, recv_buf);
        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(recv_buf);

        auto event = ccl::allreduce(send_buf,
                                    recv_buf,
                                    integral_cast<int>(count),
                                    make_oneccl_data_type(dtype),
                                    make_oneccl_reduce_op(op),
                                    device_comm_->get_ref(),
                                    stream_->get_ref());
        return new oneccl_request_impl{ std::move(event) };
    }

private:
    void init() {
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count_);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        if (rank_ == 0) {
            kvs_ = ccl::create_main_kvs();
            main_addr_ = kvs_->get_address();
            MPI_Bcast((void*)main_addr_.data(), main_addr_.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Bcast((void*)main_addr_.data(), main_addr_.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
            kvs_ = ccl::create_kvs(main_addr_);
        }
    }
    std::unique_ptr<oneccl_comm_wrapper> host_comm_;
    std::unique_ptr<oneccl_comm_wrapper> device_comm_;
    std::unique_ptr<oneccl_stream_wrapper> stream_;
    sycl::queue queue_;
    ccl::shared_ptr_class<ccl::kvs> kvs_;
    ccl::kvs::address_type main_addr_;
    std::int64_t default_root_;
    int rank_ = -1;
    int rank_count_ = -1;
};

class oneccl_communicator : public spmd_communicator {
public:
    explicit oneccl_communicator(const sycl::queue& queue, std::int64_t default_root = 0)
            : spmd_communicator(new oneccl_communicator_impl{ queue, default_root }) {}
};

} // namespace v1

using v1::oneccl_communicator;

} // namespace oneapi::dal::detail

#endif
