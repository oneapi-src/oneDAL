/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/communicator_utils.hpp"

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

    std::int64_t get_root_rank() override {
        return root_rank;
    }

    std::int64_t get_rank_count() override {
        return rank_count;
    }

    void barrier() override {}

    request_t* bcast(byte_t* send_buf, std::int64_t count, std::int64_t root) override {
        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* bcast(sycl::queue& q,
                     byte_t* send_buf,
                     std::int64_t count,
                     std::int64_t root) override {
        return nullptr;
    }
#endif

    request_t* gather(const byte_t* send_buf,
                      std::int64_t send_count,
                      byte_t* recv_buf,
                      std::int64_t recv_count,
                      std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        // TODO: Copy from `send_buf` to `recv_buf` if `recv_buf != send_buf`
        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* gather(sycl::queue& q,
                      const byte_t* send_buf,
                      std::int64_t send_count,
                      byte_t* recv_buf,
                      std::int64_t recv_count,
                      std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        // TODO: Copy from `send_buf` to `recv_buf` if `recv_buf != send_buf`
        return nullptr;
    }

#endif

    request_t* gatherv(const byte_t* send_buf,
                       std::int64_t send_count,
                       byte_t* recv_buf,
                       const std::int64_t* recv_count,
                       const std::int64_t* displs,
                       std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        // TODO: Copy from `send_buf` to `recv_buf` if `recv_buf != send_buf`
        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* gatherv(sycl::queue& q,
                       const byte_t* send_buf,
                       std::int64_t send_count,
                       byte_t* recv_buf,
                       const std::int64_t* recv_count,
                       const std::int64_t* displs,
                       std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        // TODO: Copy from `send_buf` to `recv_buf` if `recv_buf != send_buf`
        return nullptr;
    }
#endif

    request_t* allreduce(const byte_t* send_buf,
                         byte_t* recv_buf,
                         std::int64_t count,
                         const data_type& dtype,
                         const dal::detail::spmd_reduce_op& op) override {
        // TODO: Copy from `send_buf` to `recv_buf` if `recv_buf != send_buf`
        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* allreduce(sycl::queue& q,
                         const byte_t* send_buf,
                         byte_t* recv_buf,
                         std::int64_t count,
                         const data_type& dtype,
                         const dal::detail::spmd_reduce_op& op) override {
        // TODO: Copy from `send_buf` to `recv_buf` if `recv_buf != send_buf`
        return nullptr;
    }
#endif

    request_t* allgather(const byte_t* send_buf,
                         std::int64_t send_count,
                         byte_t* recv_buf,
                         std::int64_t recv_count) override {
        // TODO: Copy from `send_buf` to `recv_buf` if `recv_buf != send_buf`
        return nullptr;
    }

#ifdef ONEDAL_DATA_PARALLEL
    request_t* allgather(sycl::queue& q,
                         const byte_t* send_buf,
                         std::int64_t send_count,
                         byte_t* recv_buf,
                         std::int64_t recv_count) override {
        // TODO: Copy from `send_buf` to `recv_buf` if `recv_buf != send_buf`
        return nullptr;
    }
#endif
};

class fake_spmd_communicator : public dal::detail::spmd_communicator {
public:
    using base_t = dal::detail::spmd_communicator;
    fake_spmd_communicator() : base_t(new fake_spmd_communicator_impl{}) {}
};

class spmd_request {
public:
    spmd_request() = default;
    explicit spmd_request(const dal::detail::spmd_request& req) : req_(req) {}
    explicit spmd_request(dal::detail::spmd_request&& req) : req_(std::move(req)) {}

    void wait() {
        req_.wait();
    }

    bool test() {
        return req_.test();
    }

private:
    dal::detail::spmd_request req_;
};

class spmd_communicator {
public:
    spmd_communicator() : comm_(fake_spmd_communicator{}), is_distributed_(false) {}

    explicit spmd_communicator(const dal::detail::spmd_communicator& comm)
            : comm_(comm),
              is_distributed_(true) {}

    bool is_distributed() const {
        return is_distributed_;
    }

    std::int64_t get_rank() const {
        return comm_.get_rank();
    }

    std::int64_t get_rank_count() const {
        return comm_.get_rank_count();
    }

    std::int64_t get_root_rank() const {
        return comm_.get_root_rank();
    }

    bool is_root() const {
        return dal::detail::is_root_rank(comm_);
    }

    template <typename IfBody>
    auto if_root(IfBody&& if_body) const {
        return dal::detail::if_root_rank(comm_, std::forward<IfBody>(if_body));
    }

    template <typename IfBody, typename ElseBody>
    auto if_root_else(IfBody&& if_body, ElseBody&& else_body) const {
        return dal::detail::if_root_rank_else(comm_,
                                              std::forward<IfBody>(if_body),
                                              std::forward<ElseBody>(else_body));
    }

    template <typename T>
    void bcast(T* buf, std::int64_t count, std::int64_t root) const {
        dal::detail::bcast(comm_, buf, count, root);
    }

    template <typename T>
    void bcast(T* buf, std::int64_t count) const {
        dal::detail::bcast(comm_, buf, count);
    }

    template <typename T>
    void bcast(T& value) const {
        dal::detail::bcast(comm_, value);
    }

    template <typename T>
    void gather(const T* send_buf,
                std::int64_t send_count,
                T* recv_buf,
                std::int64_t recv_count,
                std::int64_t root) const {
        dal::detail::gather(comm_, send_buf, send_count, recv_buf, recv_count, root);
    }

    template <typename T>
    void gather(const T* send_buf,
                std::int64_t send_count,
                T* recv_buf,
                std::int64_t recv_count) const {
        dal::detail::gather(comm_, send_buf, send_count, recv_buf, recv_count);
    }

    template <typename T>
    void gather(const T* send_buf, std::int64_t send_count, T* recv_buf) const {
        dal::detail::gather(comm_, send_buf, send_count, recv_buf);
    }

    template <typename T>
    std::vector<T> gather(const T& send) const {
        return dal::detail::gather(comm_, send);
    }

    template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
    spmd_request allreduce(T& scalar) const {
        return spmd_request{ dal::detail::allreduce(comm_, scalar) };
    }

    template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
    spmd_request allreduce(const array<T>& ary) const {
        return spmd_request{ dal::detail::allreduce(comm_, ary) };
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    spmd_request allreduce(const array<T>& ary, const event_vector& deps) const {
        // TODO: Pass `deps` to allreduce free function
        sycl::event::wait_and_throw(deps);
        return spmd_request{ dal::detail::allreduce(comm_, ary) };
    }
#endif

    template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
    spmd_request allgather(const array<T>& send_ary, const array<T>& recv_ary) const {
        return spmd_request{ dal::detail::allgather(comm_, send_ary, recv_ary) };
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
    spmd_request allgather(const array<T>& send_ary,
                           const array<T>& recv_ary,
                           const event_vector& deps) const {
        // TODO: Pass `deps` to allreduce free function
        sycl::event::wait_and_throw(deps);
        return spmd_request{ dal::detail::allgather(comm_, send_ary, recv_ary) };
    }
#endif

private:
    dal::detail::spmd_communicator comm_;
    bool is_distributed_;
};

} // namespace oneapi::dal::backend
