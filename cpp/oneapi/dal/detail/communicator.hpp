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

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class communication_error : public runtime_error, public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

    const char* what() const noexcept override {
        return std::runtime_error::what();
    }
};

class spmd_request_iface {
public:
    virtual ~spmd_request_iface() = default;
    virtual void wait() = 0;
    virtual bool test() = 0;
};

enum class spmd_reduce_op {
    sum,
};

class spmd_communicator_iface {
public:
    virtual ~spmd_communicator_iface() = default;

    virtual std::int64_t get_rank() = 0;
    virtual std::int64_t get_root_rank() = 0;
    virtual std::int64_t get_rank_count() = 0;

    virtual void barrier() = 0;

    virtual spmd_request_iface* bcast(byte_t* send_buf, std::int64_t count, std::int64_t root) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* bcast(sycl::queue& q,
                                      byte_t* send_buf,
                                      std::int64_t count,
                                      std::int64_t root) = 0;
#endif

    virtual spmd_request_iface* gather(const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       std::int64_t recv_count,
                                       std::int64_t root) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* gather(sycl::queue& q,
                                       const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       std::int64_t recv_count,
                                       std::int64_t root) = 0;

#endif

    virtual spmd_request_iface* gatherv(const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_count,
                                        const std::int64_t* displs,
                                        std::int64_t root) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* gatherv(sycl::queue& q,
                                        const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_count,
                                        const std::int64_t* displs,
                                        std::int64_t root) = 0;
#endif

    virtual spmd_request_iface* allreduce(const byte_t* send_buf,
                                          byte_t* recv_buf,
                                          std::int64_t count,
                                          const data_type& dtype,
                                          const spmd_reduce_op& op) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* allreduce(sycl::queue& q,
                                          const byte_t* send_buf,
                                          byte_t* recv_buf,
                                          std::int64_t count,
                                          const data_type& dtype,
                                          const spmd_reduce_op& op) = 0;
#endif

    virtual spmd_request_iface* allgather(const byte_t* send_buf,
                                          std::int64_t send_count,
                                          byte_t* recv_buf,
                                          std::int64_t recv_count) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* allgather(sycl::queue& q,
                                          const byte_t* send_buf,
                                          std::int64_t send_count,
                                          byte_t* recv_buf,
                                          std::int64_t recv_count) = 0;
#endif
};

class spmd_request : public base {
    friend dal::detail::pimpl_accessor;

public:
    spmd_request() = default;

    void wait() {
        if (impl_) {
            return impl_->wait();
        }
    }

    bool test() {
        if (impl_) {
            return impl_->test();
        }
        return true;
    }

private:
    explicit spmd_request(spmd_request_iface* impl) : impl_(impl) {}
    dal::detail::pimpl<spmd_request_iface> impl_;
};

class spmd_communicator : public base {
public:
    std::int64_t get_rank() const {
        // TODO: Handle null impl_
        return impl_->get_rank();
    }

    std::int64_t get_rank_count() const {
        // TODO: Handle null impl_
        return impl_->get_rank_count();
    }

    std::int64_t get_root_rank() const {
        // TODO: Handle null impl_
        return impl_->get_root_rank();
    }

    void barrier() const {
        // TODO: Handle null impl_
        return impl_->barrier();
    }

    /// Broadcasts a message from the `root` rank to all other ranks
    ///
    /// @param send_buff
    /// @param count
    /// @param root
    /// @return The object to track the progress of the operation
    spmd_request bcast(byte_t* send_buf, std::int64_t count, std::int64_t root) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(impl_->bcast(send_buf, count, root));
    }

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request bcast(sycl::queue& q,
                       byte_t* send_buf,
                       std::int64_t count,
                       std::int64_t root) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(impl_->bcast(q, send_buf, count, root));
    }
#endif

    /// Collects data from all the ranks within a communicator into a single buffer
    ///
    /// @param send_buff  The send buffer
    /// @param send_count The number of elements in `send_buff`
    /// @param recv_buf   The receiveing buffer, must contain at least
    ///                   `rank_count * recv_count` elements, significant only at `root`
    /// @param recv_count The number of elements received from each rank, significant only at `root`
    /// @param root       The rank of receiving process
    /// @return The object to track the progress of the operation
    spmd_request gather(const byte_t* send_buf,
                        std::int64_t send_count,
                        byte_t* recv_buf,
                        std::int64_t recv_count,
                        std::int64_t root) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(
            impl_->gather(send_buf, send_count, recv_buf, recv_count, root));
    }

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request gather(sycl::queue& q,
                        const byte_t* send_buf,
                        std::int64_t send_count,
                        byte_t* recv_buf,
                        std::int64_t recv_count,
                        std::int64_t root) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(
            impl_->gather(q, send_buf, send_count, recv_buf, recv_count, root));
    }
#endif

    spmd_request gatherv(const byte_t* send_buf,
                         std::int64_t send_count,
                         byte_t* recv_buf,
                         const std::int64_t* recv_count,
                         const std::int64_t* displs,
                         std::int64_t root) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(
            impl_->gatherv(send_buf, send_count, recv_buf, recv_count, displs, root));
    }

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request gatherv(sycl::queue& q,
                         const byte_t* send_buf,
                         std::int64_t send_count,
                         byte_t* recv_buf,
                         const std::int64_t* recv_count,
                         const std::int64_t* displs,
                         std::int64_t root) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(
            impl_->gatherv(q, send_buf, send_count, recv_buf, recv_count, displs, root));
    }
#endif

    spmd_request allreduce(const byte_t* send_buf,
                           byte_t* recv_buf,
                           std::int64_t count,
                           const data_type& dtype,
                           const spmd_reduce_op& op) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(
            impl_->allreduce(send_buf, recv_buf, count, dtype, op));
    }

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request allreduce(sycl::queue& q,
                           const byte_t* send_buf,
                           byte_t* recv_buf,
                           std::int64_t count,
                           const data_type& dtype,
                           const spmd_reduce_op& op) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(
            impl_->allreduce(q, send_buf, recv_buf, count, dtype, op));
    }
#endif

    spmd_request allgather(const byte_t* send_buf,
                           std::int64_t send_count,
                           byte_t* recv_buf,
                           std::int64_t recv_count) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(
            impl_->allgather(send_buf, send_count, recv_buf, recv_count));
    }

#ifdef ONEDAL_DATA_PARALLEL
    spmd_request allgather(sycl::queue& q,
                           const byte_t* send_buf,
                           std::int64_t send_count,
                           byte_t* recv_buf,
                           std::int64_t recv_count) const {
        // TODO: Handle null impl_
        return dal::detail::make_private<spmd_request>(
            impl_->allgather(q, send_buf, send_count, recv_buf, recv_count));
    }
#endif

protected:
    explicit spmd_communicator(spmd_communicator_iface* impl) : impl_(impl) {}

    template <typename Impl>
    Impl& get_impl() const {
        // TODO: Handle null impl_
        return static_cast<Impl&>(*impl_);
    }

private:
    dal::detail::pimpl<spmd_communicator_iface> impl_;
};

} // namespace v1

using v1::communication_error;
using v1::spmd_reduce_op;
using v1::spmd_request_iface;
using v1::spmd_communicator_iface;
using v1::spmd_request;
using v1::spmd_communicator;

} // namespace oneapi::dal::detail
