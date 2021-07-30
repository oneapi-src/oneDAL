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

#include "oneapi/dal/detail/communicator.hpp"

namespace oneapi::dal::detail::v1 {

const char* communication_error::what() const noexcept {
    return std::runtime_error::what();
}

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
    return make_private<spmd_request>(impl_->bcast(send_buf, count, dtype, root));
}

#ifdef ONEDAL_DATA_PARALLEL
spmd_request spmd_communicator::bcast(sycl::queue& q,
                                      byte_t* send_buf,
                                      std::int64_t count,
                                      const data_type& dtype,
                                      std::int64_t root) const {
    return make_private<spmd_request>(impl_->bcast(q, send_buf, count, dtype, root));
}
#endif

spmd_request spmd_communicator::gather(const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       std::int64_t recv_count,
                                       const data_type& dtype,
                                       std::int64_t root) const {
    return make_private<spmd_request>(
        impl_->gather(send_buf, send_count, recv_buf, recv_count, dtype, root));
}

#ifdef ONEDAL_DATA_PARALLEL
spmd_request spmd_communicator::gather(sycl::queue& q,
                                       const byte_t* send_buf,
                                       std::int64_t send_count,
                                       byte_t* recv_buf,
                                       std::int64_t recv_count,
                                       const data_type& dtype,
                                       std::int64_t root) const {
    return make_private<spmd_request>(
        impl_->gather(q, send_buf, send_count, recv_buf, recv_count, dtype, root));
}
#endif

spmd_request spmd_communicator::gatherv(const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_count,
                                        const std::int64_t* displs,
                                        const data_type& dtype,
                                        std::int64_t root) const {
    return make_private<spmd_request>(
        impl_->gatherv(send_buf, send_count, recv_buf, recv_count, displs, dtype, root));
}

#ifdef ONEDAL_DATA_PARALLEL
spmd_request spmd_communicator::gatherv(sycl::queue& q,
                                        const byte_t* send_buf,
                                        std::int64_t send_count,
                                        byte_t* recv_buf,
                                        const std::int64_t* recv_count,
                                        const std::int64_t* displs,
                                        const data_type& dtype,
                                        std::int64_t root) const {
    return make_private<spmd_request>(
        impl_->gatherv(q, send_buf, send_count, recv_buf, recv_count, displs, dtype, root));
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
                                          const spmd_reduce_op& op) const {
    return make_private<spmd_request>(impl_->allreduce(q, send_buf, recv_buf, count, dtype, op));
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
                                          const data_type& dtype) const {
    return make_private<spmd_request>(
        impl_->allgather(q, send_buf, send_count, recv_buf, recv_count, dtype));
}
#endif

} // namespace oneapi::dal::detail::v1
