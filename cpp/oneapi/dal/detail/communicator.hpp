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

#ifdef ONEDAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif

#include "oneapi/dal/array.hpp"

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
    virtual std::int64_t get_rank_count() = 0;
    virtual std::int64_t get_default_root_rank() = 0;

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
                                      const std::vector<sycl::event>& deps,
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
                                       const std::vector<sycl::event>& deps,
                                       std::int64_t root) = 0;

#endif

    virtual spmd_request_iface* gatherv(const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_counts,
                                        const std::int64_t* displs,
                                        const data_type& dtype,
                                        std::int64_t root) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual spmd_request_iface* gatherv(sycl::queue& q,
                                        const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_counts_host,
                                        const std::int64_t* displs_host,
                                        const data_type& dtype,
                                        const std::vector<sycl::event>& deps,
                                        std::int64_t root) = 0;
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
                                          const data_type& dtype,
                                          const std::vector<sycl::event>& deps) = 0;
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
                                          const spmd_reduce_op& op,
                                          const std::vector<sycl::event>& deps) = 0;
#endif
};

/// The class that tracks progress of async SPMD communicator operation
class spmd_request : public base {
    friend dal::detail::pimpl_accessor;

public:
    /// Creates completed request that does not track progress of any operation
    spmd_request();

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
private:
    template <typename T>
    static constexpr bool is_primitive_v = std::is_arithmetic_v<T>;

    template <typename T>
    using enable_if_primitive_t = std::enable_if_t<is_primitive_v<T>>;

public:
    spmd_communicator() = delete;

    /// Returns id of the current rank
    std::int64_t get_rank() const;

    /// Returns the total number of active ranks
    std::int64_t get_rank_count() const;

    /// Returns id of the rank considered as a root for collective operations
    std::int64_t get_default_root_rank() const;

    /// Returns `true` if the current rank is root
    bool is_root_rank(std::int64_t root = -1) const {
        return get_rank() == fix_root_rank(root);
    }

    /// Performs `if_body` only if the current rank is root, passes through the
    /// result returned by the `if_body`. The result type must be trivially-constructable.
    template <typename IfBody>
    auto if_root_rank(IfBody&& if_body, std::int64_t root = -1) const {
        if (is_root_rank(root)) {
            return if_body();
        }
        else {
            using return_t = decltype(if_body());
            return return_t{};
        }
    }

    /// Performs `if_body` only if the current rank is root, otherwise performs
    /// `else_body`. Passes through the result returned by the `if_body` or
    /// `else_body`. The `if_body` and `else_body` must return value of the same type.
    template <typename IfBody, typename ElseBody>
    auto if_root_rank_else(IfBody&& if_body, ElseBody&& else_body, std::int64_t root = -1) {
        if (is_root_rank(root)) {
            return if_body();
        }
        else {
            return else_body();
        }
    }

    /// Blocks until all ranks in the communicator have reached this function
    void barrier() const;

    /// Broadcasts a message from the `root` rank to all other ranks
    ///
    /// @param buf   The buffer which content is broadcasted
    /// @param count The number of elements of `dtype` in `send_buf`
    /// @param dtype The type of elements in the passed buffers
    /// @param root  The rank of the broadcasting process, if the passed
    ///              rank is negative, the default root rank is used
    ///
    /// @return The object to track the progress of the operation
    spmd_request bcast(byte_t* buf,
                       std::int64_t count,
                       const data_type& dtype,
                       std::int64_t root = -1) const;

#ifdef ONEDAL_DATA_PARALLEL
    /// `bcast` that accepts USM pointers
    spmd_request bcast(sycl::queue& q,
                       byte_t* buf,
                       std::int64_t count,
                       const data_type& dtype,
                       const std::vector<sycl::event>& deps = {},
                       std::int64_t root = -1) const;
#endif

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request bcast(T* buf, std::int64_t count, std::int64_t root = -1) const {
        return bcast(reinterpret_cast<byte_t*>(buf), count, make_data_type<T>(), root);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request bcast(sycl::queue& q,
                       T* buf,
                       std::int64_t count,
                       const std::vector<sycl::event>& deps = {},
                       std::int64_t root = -1) const {
        return bcast(q, reinterpret_cast<byte_t*>(buf), count, make_data_type<T>(), deps, root);
    }
#endif

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request bcast(T& value, std::int64_t root = -1) const {
        return bcast(&value, 1, root);
    }

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request bcast(const array<T>& ary, std::int64_t root = -1) const {
        std::int64_t count = if_root_rank(
            [&]() {
                return ary.get_count();
            },
            root);

        bcast(count, root).wait();
        ONEDAL_ASSERT(ary.get_count() >= count);

        spmd_request request;
        if (is_root_rank(root)) {
            // `const_cast` is safe here, `bcast` called on the
            // root rank does not modify the values
            __ONEDAL_IF_QUEUE__(ary.get_queue(), {
                auto q = ary.get_queue().value();
                request = bcast(q, const_cast<T*>(ary.get_data()), count, {}, root);
            });

            __ONEDAL_IF_NO_QUEUE__(ary.get_queue(), { //
                request = bcast(const_cast<T*>(ary.get_data()), count, root);
            });
        }
        else {
            ONEDAL_ASSERT(ary.has_mutable_data());

            __ONEDAL_IF_QUEUE__(ary.get_queue(), {
                auto q = ary.get_queue().value();
                request = bcast(q, ary.get_mutable_data(), count, {}, root);
            });

            __ONEDAL_IF_NO_QUEUE__(ary.get_queue(), { //
                request = bcast(ary.get_mutable_data(), count, root);
            });
        }

        return request;
    }

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
                        std::int64_t root = -1) const;

#ifdef ONEDAL_DATA_PARALLEL
    /// `gather` that accepts USM pointers
    spmd_request gather(sycl::queue& q,
                        const byte_t* send_buf,
                        std::int64_t send_count,
                        byte_t* recv_buf,
                        std::int64_t recv_count,
                        const data_type& dtype,
                        const std::vector<sycl::event>& deps = {},
                        std::int64_t root = -1) const;
#endif

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request gather(const T* send_buf,
                        std::int64_t send_count,
                        T* recv_buf,
                        std::int64_t recv_count,
                        std::int64_t root = -1) const {
        return gather(reinterpret_cast<const byte_t*>(send_buf),
                      send_count,
                      reinterpret_cast<byte_t*>(recv_buf),
                      recv_count,
                      make_data_type<T>(),
                      root);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request gather(sycl::queue& q,
                        const T* send_buf,
                        std::int64_t send_count,
                        T* recv_buf,
                        std::int64_t recv_count,
                        const std::vector<sycl::event>& deps = {},
                        std::int64_t root = -1) const {
        return gather(q,
                      reinterpret_cast<const byte_t*>(send_buf),
                      send_count,
                      reinterpret_cast<byte_t*>(recv_buf),
                      recv_count,
                      make_data_type<T>(),
                      deps,
                      root);
    }
#endif

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request gather(const T& send, T* recv, const std::int64_t root = -1) const {
        return gather(&send, 1, recv, 1, root);
    }

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    array<T> gather(const T& send, const std::int64_t root = -1) const {
        array<T> recv;
        T* recv_ptr = nullptr;

        if (is_root_rank(root)) {
            recv.reset(get_rank_count());
            recv_ptr = recv.get_mutable_data();
        }

        gather(send, recv_ptr, root).wait();

        return recv;
    }

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request gather(const array<T>& send,
                        const array<T>& recv,
                        const std::int64_t root = -1) const {
#ifdef ONEDAL_ENABLE_ASSERT
        check_if_same_send_count(send.get_count(), root);

        // Check if recv allocated count is enough to store all the received values
        // Note: `recv` must be allocated at least on root node
        const std::int64_t min_recv_count = get_min_recv_count(send.get_count());
        if (is_root_rank(root)) {
            ONEDAL_ASSERT(recv.get_count() >= min_recv_count);
        }
#endif

        if (send.get_count() == 0) {
            if (is_root_rank(root)) {
                ONEDAL_ASSERT(recv.get_count() == 0);
            }
            return spmd_request{};
        }

        T* recv_ptr = nullptr;
        if (is_root_rank(root)) {
            ONEDAL_ASSERT(recv.has_mutable_data());
            recv_ptr = recv.get_mutable_data();
        }

        spmd_request request;

        __ONEDAL_IF_QUEUE__(send.get_queue(), {
            auto q = send.get_queue().value();

            if (is_root_rank(root)) {
                ONEDAL_ASSERT(recv.get_queue().has_value());
                ONEDAL_ASSERT(recv.get_queue().value().get_context() == q.get_context());
            }

            request = gather( //
                q,
                send.get_data(),
                send.get_count(),
                recv_ptr,
                send.get_count(),
                {},
                root);
        });

        __ONEDAL_IF_NO_QUEUE__(send.get_queue(), {
            request = gather( //
                send.get_data(),
                send.get_count(),
                recv_ptr,
                send.get_count(),
                root);
        });

        return request;
    }

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
                         const std::int64_t* recv_counts,
                         const std::int64_t* displs,
                         const data_type& dtype,
                         std::int64_t root = -1) const;

#ifdef ONEDAL_DATA_PARALLEL
    /// `gatherv` that accepts USM pointers
    /// `recv_count_host` must be host-allocated memory!
    /// `displs_host` must be host-allocated memory!
    spmd_request gatherv(sycl::queue& q,
                         const byte_t* send_buf,
                         std::int64_t send_count,
                         byte_t* recv_buf,
                         const std::int64_t* recv_counts_host,
                         const std::int64_t* displs_host,
                         const data_type& dtype,
                         const std::vector<sycl::event>& deps = {},
                         std::int64_t root = -1) const;
#endif

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request gatherv(const T* send_buf,
                         std::int64_t send_count,
                         T* recv_buf,
                         const std::int64_t* recv_counts,
                         const std::int64_t* displs,
                         std::int64_t root = -1) const {
        return gatherv(reinterpret_cast<const byte_t*>(send_buf),
                       send_count,
                       reinterpret_cast<byte_t*>(recv_buf),
                       recv_counts,
                       displs,
                       make_data_type<T>(),
                       root);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request gatherv(sycl::queue& q,
                         const T* send_buf,
                         std::int64_t send_count,
                         T* recv_buf,
                         const std::int64_t* recv_counts,
                         const std::int64_t* displs,
                         const std::vector<sycl::event>& deps = {},
                         std::int64_t root = -1) const {
        return gatherv(q,
                       reinterpret_cast<const byte_t*>(send_buf),
                       send_count,
                       reinterpret_cast<byte_t*>(recv_buf),
                       recv_counts,
                       displs,
                       make_data_type<T>(),
                       deps,
                       root);
    }
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
                           const data_type& dtype,
                           const std::vector<sycl::event>& deps = {}) const;
#endif

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request allgather(const T* send_buf,
                           std::int64_t send_count,
                           T* recv_buf,
                           std::int64_t recv_count) const {
        return allgather(reinterpret_cast<const byte_t*>(send_buf),
                         send_count,
                         reinterpret_cast<byte_t*>(recv_buf),
                         recv_count,
                         make_data_type<T>());
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request allgather(sycl::queue& q,
                           const T* send_buf,
                           std::int64_t send_count,
                           T* recv_buf,
                           std::int64_t recv_count,
                           const std::vector<sycl::event>& deps = {}) const {
        return allgather(q,
                         reinterpret_cast<const byte_t*>(send_buf),
                         send_count,
                         reinterpret_cast<byte_t*>(recv_buf),
                         recv_count,
                         make_data_type<T>(),
                         deps);
    }
#endif

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request allgather(const array<T>& send, const array<T>& recv) const {
#ifdef ONEDAL_ENABLE_ASSERT
        check_if_same_send_count(send.get_count(), get_default_root_rank());

        // Check if recv allocated count is enough to store all the received values
        const std::int64_t min_recv_count = get_min_recv_count(send.get_count());
        ONEDAL_ASSERT(recv.get_count() >= min_recv_count);
#endif

        if (send.get_count() == 0) {
            ONEDAL_ASSERT(recv.get_count() == 0);
            return spmd_request{};
        }

        ONEDAL_ASSERT(send.get_count() > 0);
        ONEDAL_ASSERT(recv.has_mutable_data());

        spmd_request request;

        __ONEDAL_IF_QUEUE__(send.get_queue(), {
            auto q = send.get_queue().value();

            ONEDAL_ASSERT(recv.get_queue().has_value());
            ONEDAL_ASSERT(recv.get_queue().value().get_context() == q.get_context());

            request = allgather(q,
                                send.get_data(),
                                send.get_count(),
                                recv.get_mutable_data(),
                                send.get_count());
        });

        __ONEDAL_IF_NO_QUEUE__(send.get_queue(), {
            request = allgather(send.get_data(),
                                send.get_count(),
                                recv.get_mutable_data(),
                                send.get_count());
        });

        return request;
    }

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
                           const spmd_reduce_op& op = spmd_reduce_op::sum) const;

#ifdef ONEDAL_DATA_PARALLEL
    /// `allreduce` that accepts USM pointers
    spmd_request allreduce(sycl::queue& q,
                           const byte_t* send_buf,
                           byte_t* recv_buf,
                           std::int64_t count,
                           const data_type& dtype,
                           const spmd_reduce_op& op = spmd_reduce_op::sum,
                           const std::vector<sycl::event>& deps = {}) const;
#endif

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request allreduce(const T* send_buf,
                           T* recv_buf,
                           std::int64_t count,
                           const spmd_reduce_op& op = spmd_reduce_op::sum) const {
        return allreduce(reinterpret_cast<const byte_t*>(send_buf),
                         reinterpret_cast<byte_t*>(recv_buf),
                         count,
                         make_data_type<T>(),
                         op);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request allreduce(sycl::queue& q,
                           const T* send_buf,
                           T* recv_buf,
                           std::int64_t count,
                           const spmd_reduce_op& op = spmd_reduce_op::sum,
                           const std::vector<sycl::event>& deps = {}) const {
        return allreduce(q,
                         reinterpret_cast<const byte_t*>(send_buf),
                         reinterpret_cast<byte_t*>(recv_buf),
                         count,
                         make_data_type<T>(),
                         op,
                         deps);
    }
#endif

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request allreduce(T& scalar, const spmd_reduce_op& op = spmd_reduce_op::sum) const {
        return allreduce(&scalar, &scalar, 1, op);
    }

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    spmd_request allreduce(const array<T>& ary,
                           const spmd_reduce_op& op = spmd_reduce_op::sum) const {
#ifdef ONEDAL_ENABLE_ASSERT
        check_if_same_send_count(ary.get_count(), get_default_root_rank());
#endif

        if (ary.get_count() == 0) {
            return spmd_request{};
        }

        ONEDAL_ASSERT(ary.get_count() > 0);
        ONEDAL_ASSERT(ary.has_mutable_data());

        spmd_request request;

        __ONEDAL_IF_QUEUE__(ary.get_queue(), {
            auto q = ary.get_queue().value();
            request = allreduce(q, ary.get_data(), ary.get_mutable_data(), ary.get_count(), op);
        });

        __ONEDAL_IF_NO_QUEUE__(ary.get_queue(), {
            request = allreduce(ary.get_data(), ary.get_mutable_data(), ary.get_count(), op);
        });

        return request;
    }

protected:
    explicit spmd_communicator(spmd_communicator_iface* impl) : impl_(impl) {}

    template <typename Impl>
    Impl& get_impl() const {
        static_assert(std::is_base_of_v<spmd_communicator_iface, Impl>);
        return static_cast<Impl&>(*impl_);
    }

    std::int64_t fix_root_rank(std::int64_t root) const {
        if (root < 0) {
            return get_default_root_rank();
        }
        return root;
    }

#ifdef ONEDAL_ENABLE_ASSERT
    void check_if_same_send_count(std::int64_t send_count, std::int64_t root) const {
        const auto counts = gather(send_count, root);
        if (is_root_rank(root)) {
            ONEDAL_ASSERT(counts.get_count() > 0);
            for (std::int64_t i = 0; i < counts.get_count(); i++) {
                ONEDAL_ASSERT(counts[i] == counts[0]);
            }
        }
    }
#endif

    std::int64_t get_min_recv_count(std::int64_t send_count) const {
        std::int64_t minimal_recv_count = send_count;
        allreduce(minimal_recv_count).wait();
        return minimal_recv_count;
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
