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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/communicator.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/array.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::detail::v1 {

#ifdef ONEDAL_DATA_PARALLEL
static void wait_request(spmd::request_iface* request) {
    if (request != nullptr) {
        request->wait();
    }
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
spmd::request_iface* spmd_communicator_via_host_impl::bcast(sycl::queue& q,
                                                            byte_t* send_buf,
                                                            std::int64_t count,
                                                            const data_type& dtype,
                                                            const std::vector<sycl::event>& deps,
                                                            std::int64_t root) {
    ONEDAL_ASSERT(root >= 0);

    if (count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(count > 0);

    preview::detail::check_if_pointer_matches_queue(q, send_buf);
    sycl::event::wait_and_throw(deps);
    {
        ONEDAL_PROFILER_TASK(comm.bcast_gpu, q);
        wait_request(bcast(send_buf, count, dtype, root));
    }

    return nullptr;
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
spmd::request_iface* spmd_communicator_via_host_impl::allgatherv(
    sycl::queue& q,
    const byte_t* send_buf,
    std::int64_t send_count,
    byte_t* recv_buf,
    const std::int64_t* recv_counts_host,
    const std::int64_t* displs_host,
    const data_type& dtype,
    const std::vector<sycl::event>& deps) {
    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(recv_buf);
    ONEDAL_ASSERT(recv_counts_host);
    ONEDAL_ASSERT(displs_host);

    preview::detail::check_if_pointer_matches_queue(q, send_buf);
    preview::detail::check_if_pointer_matches_queue(q, recv_buf);
    sycl::event::wait_and_throw(deps);

    {
        ONEDAL_PROFILER_TASK(comm.allgatherv_gpu, q);
        wait_request(
            allgatherv(send_buf, send_count, recv_buf, recv_counts_host, displs_host, dtype));
    }
    return nullptr;
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
spmd::request_iface* spmd_communicator_via_host_impl::allreduce(
    sycl::queue& q,
    const byte_t* send_buf,
    byte_t* recv_buf,
    std::int64_t count,
    const data_type& dtype,
    const spmd::reduce_op& op,
    const std::vector<sycl::event>& deps) {
    if (count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(send_buf);
    ONEDAL_ASSERT(recv_buf);
    ONEDAL_ASSERT(count > 0);

    preview::detail::check_if_pointer_matches_queue(q, send_buf);
    preview::detail::check_if_pointer_matches_queue(q, recv_buf);
    sycl::event::wait_and_throw(deps);

    {
        ONEDAL_PROFILER_TASK(comm.allreduce_gpu, q);
        wait_request(allreduce(send_buf, recv_buf, count, dtype, op));
    }

    return nullptr;
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
spmd::request_iface* spmd_communicator_via_host_impl::sendrecv_replace(
    sycl::queue& q,
    byte_t* buf,
    std::int64_t count,
    const data_type& dtype,
    std::int64_t destination_rank,
    std::int64_t source_rank,
    const std::vector<sycl::event>& deps) {
    ONEDAL_ASSERT(destination_rank >= 0);
    ONEDAL_ASSERT(source_rank >= 0);

    if (count == 0) {
        return nullptr;
    }

    ONEDAL_ASSERT(buf);
    ONEDAL_ASSERT(count > 0);

    preview::detail::check_if_pointer_matches_queue(q, buf);
    sycl::event::wait_and_throw(deps);

    {
        ONEDAL_PROFILER_TASK(comm.srr_gpu, q);
        wait_request(sendrecv_replace(buf, count, dtype, destination_rank, source_rank));
    }

    return nullptr;
}
#endif

} // namespace oneapi::dal::detail::v1
