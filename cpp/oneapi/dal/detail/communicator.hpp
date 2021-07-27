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

#ifdef ONEDAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class communication_error : public runtime_error, public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    const char* what() const noexcept override;
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

    virtual spmd_request_iface* bcast(byte_t* send_buf,
                                      std::int64_t count,
                                      const data_type& dtype,
                                      std::int64_t root) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* bcast(sycl::queue& q,
                                      byte_t* send_buf,
                                      std::int64_t count,
                                      const data_type& dtype,
                                      std::int64_t root) = 0;
#endif

    virtual spmd_request_iface* gather(const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       std::int64_t recv_count,
                                       const data_type& dtype,
                                       std::int64_t root) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* gather(sycl::queue& q,
                                       const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       std::int64_t recv_count,
                                       const data_type& dtype,
                                       std::int64_t root) = 0;

#endif

    virtual spmd_request_iface* gatherv(const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_count,
                                        const std::int64_t* displs,
                                        const data_type& dtype,
                                        std::int64_t root) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* gatherv(sycl::queue& q,
                                        const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_count,
                                        const std::int64_t* displs,
                                        const data_type& dtype,
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
                                          std::int64_t recv_count,
                                          const data_type& dtype) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* allgather(sycl::queue& q,
                                          const byte_t* send_buf,
                                          std::int64_t send_count,
                                          byte_t* recv_buf,
                                          std::int64_t recv_count,
                                          const data_type& dtype) = 0;
#endif
};

/// The class that tracks progress of async SPMD communicator operation
class spmd_request : public base {
    friend dal::detail::pimpl_accessor;

public:
    /// Creates completed request that does not track progress of any operation
    spmd_request() = default;

    /// Blocks until request is preformed
    void wait();

    /// Returns true if request is completed
    bool test();

private:
    explicit spmd_request(spmd_request_iface* impl) : impl_(impl) {}
    dal::detail::pimpl<spmd_request_iface> impl_;
};

/// Low-level MPI-like communicator
class spmd_communicator : public base {
public:
    spmd_communicator() = delete;

    /// Returns id of the current rank
    std::int64_t get_rank() const;

    /// Returns the total number of active ranks
    std::int64_t get_rank_count() const;

    /// Returns id of the rank considered as a root for collective operations
    std::int64_t get_root_rank() const;

    /// Blocks until all ranks in the communicator have reached this function
    void barrier() const;

    /// Broadcasts a message from the `root` rank to all other ranks
    ///
    /// @param send_buf The buffer which content is broadcasted
    /// @param count    The number of elements of `dtype` in `send_buf`
    /// @param dtype    The type of elements in the passed buffers
    /// @param root     The rank of the broadcasting process
    ///
    /// @return The object to track the progress of the operation
    spmd_request bcast(byte_t* send_buf,
                       std::int64_t count,
                       const data_type& dtype,
                       std::int64_t root) const;

#ifdef ONEDAL_DATA_PARALLEL
    /// `bcast` that accepts USM pointers
    spmd_request bcast(sycl::queue& q,
                       byte_t* send_buf,
                       std::int64_t count,
                       const data_type& dtype,
                       std::int64_t root) const;
#endif

    /// Collects data from all the ranks within a communicator into a single buffer.
    /// The data size sent by each ranks must be the same.
    ///
    /// @param send_buf   The send buffer
    /// @param send_count The number of elements of `dtype` in `send_buf`
    /// @param recv_buf   The receiveing buffer, must contain at least
    ///                   `rank_count * recv_count` elements,
    ///                   significant only at `root`
    /// @param recv_count The number of elements of `dtype` received from
    ///                   each rank, significant only at `root`
    /// @param dtype      The type of elements in the passed buffers
    /// @param root       The rank of the receiving process
    ///
    /// @return The object to track the progress of the operation
    spmd_request gather(const byte_t* send_buf,
                        std::int64_t send_count,
                        byte_t* recv_buf,
                        std::int64_t recv_count,
                        const data_type& dtype,
                        std::int64_t root) const;

#ifdef ONEDAL_DATA_PARALLEL
    /// `gather` that accepts USM pointers
    spmd_request gather(sycl::queue& q,
                        const byte_t* send_buf,
                        std::int64_t send_count,
                        byte_t* recv_buf,
                        std::int64_t recv_count,
                        const data_type& dtype,
                        std::int64_t root) const;
#endif

    /// Collects data from all the ranks within a communicator into a single buffer.
    /// The data size send by each rank may be different.
    ///
    /// @param send_buf   The send buffer
    /// @param send_count The number of elements of `dtype` in `send_buf`
    /// @param recv_buf   The receiveing buffer, must contain at least
    ///                   `rank_count * recv_count` elements,
    ///                   significant only at `root`
    /// @param recv_count The number of elements of `dtype` received from
    ///                   each rank, must contain at least `rank_count` elements,
    ///                   significant only at `root`
    /// @param displs     Entry $i$ specifies the displacement relative to
    ///                   `recv_buf` at which to place the incoming data
    ///                   from process $i$, must contain at least `rank_count`
    ///                   elements, significant only at `root`
    /// @param dtype      The type of elements in the passed buffers
    /// @param root       The rank of the receiving process
    ///
    /// @return The object to track the progress of the operation
    spmd_request gatherv(const byte_t* send_buf,
                         std::int64_t send_count,
                         byte_t* recv_buf,
                         const std::int64_t* recv_count,
                         const std::int64_t* displs,
                         const data_type& dtype,
                         std::int64_t root) const;

#ifdef ONEDAL_DATA_PARALLEL
    /// `gatherv` that accepts USM pointers
    spmd_request gatherv(sycl::queue& q,
                         const byte_t* send_buf,
                         std::int64_t send_count,
                         byte_t* recv_buf,
                         const std::int64_t* recv_count,
                         const std::int64_t* displs,
                         const data_type& dtype,
                         std::int64_t root) const;
#endif

    /// Combines data from all ranks using reduction operation and
    /// distributes the result back to all ranks
    ///
    /// @param send_buf The send buffer
    /// @param recv_buf The receiving buffer
    /// @param count    The number of elements of `dtype` sent to and
    ///                 received from each rank
    /// @param dtype    The type of elements in the passed buffers
    /// @param op       The reduction operation
    ///
    /// @return The object to track the progress of the operation
    spmd_request allreduce(const byte_t* send_buf,
                           byte_t* recv_buf,
                           std::int64_t count,
                           const data_type& dtype,
                           const spmd_reduce_op& op) const;

#ifdef ONEDAL_DATA_PARALLEL
    /// `allreduce` that accepts USM pointers
    spmd_request allreduce(sycl::queue& q,
                           const byte_t* send_buf,
                           byte_t* recv_buf,
                           std::int64_t count,
                           const data_type& dtype,
                           const spmd_reduce_op& op) const;
#endif

    /// Gathers data from all ranks and distributes the results back to all ranks
    ///
    /// @param send_buf   The send buffer
    /// @param send_count The number of elements of `dtype` in `send_buf`
    /// @param recv_buf   The receiving buffer
    /// @param recv_count The number of elements of `dtype` in `recv_buf`
    /// @param dtype      The type of elements in the passed buffers
    ///
    /// @return The object to track the progress of the operation
    spmd_request allgather(const byte_t* send_buf,
                           std::int64_t send_count,
                           byte_t* recv_buf,
                           std::int64_t recv_count,
                           const data_type& dtype) const;

#ifdef ONEDAL_DATA_PARALLEL
    /// `allgather` that accepts USM pointers
    spmd_request allgather(sycl::queue& q,
                           const byte_t* send_buf,
                           std::int64_t send_count,
                           byte_t* recv_buf,
                           std::int64_t recv_count,
                           const data_type& dtype) const;
#endif

protected:
    explicit spmd_communicator(spmd_communicator_iface* impl) : impl_(impl) {}

    template <typename Impl>
    Impl& get_impl() const {
        static_assert(std::is_base_of_v<spmd_communicator_iface, Impl>);
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
