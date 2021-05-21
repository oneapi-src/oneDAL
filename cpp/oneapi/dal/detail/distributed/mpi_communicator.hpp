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

#include <mpi.h>
#include "oneapi/dal/detail/distributed/spmd_communicator.hpp"

namespace oneapi::dal::detail {

class mpi_communicator_error : public communication_error {
public:
    explicit mpi_communicator_error(int mpi_error_code)
            : communication_error(get_message(mpi_error_code)),
              mpi_error_code_(mpi_error_code) {}

    int get_error_code() const {
        return mpi_error_code_;
    }

private:
    static std::string get_message(int mpi_error_code) {
        char error_message_buffer[MPI_MAX_ERROR_STRING];
        int error_message_length;
        MPI_Error_string(mpi_error_code, error_message_buffer, &error_message_length);
        return std::string{ error_message_buffer, error_message_buffer + error_message_length };
    }

    int mpi_error_code_;
};

class mpi_request_impl : public spmd_request_iface {
public:
    explicit mpi_request_impl(const MPI_Request& request) : mpi_request_(request) {}

    ~mpi_request_impl() {
        // TODO: Figure out how to free collective operation's request
        // https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm
        ONEDAL_ASSERT(is_completed_);
    }

    mpi_request_impl(const mpi_request_impl&) = delete;
    mpi_request_impl& operator=(const mpi_request_impl&) = delete;

    void wait() override {
        try_throw(MPI_Wait(&mpi_request_, MPI_STATUS_IGNORE));
        is_completed_ = true;
    }

    bool test() override {
        int flag;
        try_throw(MPI_Test(&mpi_request_, &flag, MPI_STATUS_IGNORE));
        is_completed_ = bool(flag != 0);
        return is_completed_;
    }

private:
    static void try_throw(int mpi_error_code) {
        if (mpi_error_code != MPI_SUCCESS) {
            throw mpi_communicator_error{ mpi_error_code };
        }
    }

    MPI_Request mpi_request_;
    bool is_completed_ = false;
};

class mpi_communicator_impl : public spmd_communicator_iface {
public:
    explicit mpi_communicator_impl(const MPI_Comm& mpi_comm = MPI_COMM_WORLD)
            : mpi_comm_(mpi_comm) {}

    std::int64_t get_rank() override {
        if (my_rank_ >= 0) {
            return my_rank_;
        }

        int rank;
        try_throw(MPI_Comm_rank(mpi_comm_, &rank));
        my_rank_ = std::int64_t(rank);
        return my_rank_;
    }

    std::int64_t get_root_rank() override {
        return root_rank_;
    }

    void set_root_rank(std::int64_t rank) {
        ONEDAL_ASSERT(rank >= 0);
        root_rank_ = rank;
    }

    std::int64_t get_rank_count() override {
        int rank_count;
        try_throw(MPI_Comm_size(mpi_comm_, &rank_count));
        return std::int64_t(rank_count);
    }

    spmd_request_iface* bcast(byte_t* send_buf, std::int64_t count, std::int64_t root) override {
        ONEDAL_ASSERT(root >= 0);
        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(count > 0);

        MPI_Request mpi_request;
        try_throw(MPI_Ibcast(send_buf,
                             dal::detail::integral_cast<int>(count),
                             MPI_BYTE,
                             dal::detail::integral_cast<int>(root),
                             mpi_comm_,
                             &mpi_request));
        return new mpi_request_impl{ mpi_request };
    }

    spmd_request_iface* gather(const byte_t* send_buf,
                               std::int64_t send_count,
                               byte_t* recv_buf,
                               std::int64_t recv_count,
                               std::int64_t root) override {
        ONEDAL_ASSERT(root >= 0);
        if (get_rank() == root) {
            ONEDAL_ASSERT(recv_buf);
            ONEDAL_ASSERT(recv_count > 0);
        }
        else {
            ONEDAL_ASSERT(send_buf);
            ONEDAL_ASSERT(send_count > 0);
        }

        MPI_Request mpi_request;
        try_throw(MPI_Igather(send_buf,
                              dal::detail::integral_cast<int>(send_count),
                              MPI_BYTE,
                              recv_buf,
                              dal::detail::integral_cast<int>(recv_count),
                              MPI_BYTE,
                              dal::detail::integral_cast<int>(root),
                              mpi_comm_,
                              &mpi_request));
        return new mpi_request_impl{ mpi_request };
    }

    spmd_request_iface* gatherv(const byte_t* send_buf,
                                std::int64_t send_count,
                                byte_t* recv_buf,
                                const std::int64_t* recv_count,
                                const std::int64_t* displs,
                                std::int64_t root) override {
        ONEDAL_ASSERT(root >= 0);
        if (get_rank() == root) {
            ONEDAL_ASSERT(recv_buf);
            ONEDAL_ASSERT(displs);
            ONEDAL_ASSERT(recv_count > 0);
        }
        else {
            ONEDAL_ASSERT(send_buf);
            ONEDAL_ASSERT(send_count > 0);
        }

        const std::int64_t rank_count = get_rank_count();
        auto recv_count_int = std::make_unique<int[]>(rank_count);
        auto displs_int = std::make_unique<int[]>(rank_count);

        std::int64_t displs_counter = 0;
        for (std::int64_t i = 0; i < rank_count; i++) {
            ONEDAL_ASSERT(recv_count[i] > 0);
            ONEDAL_ASSERT(displs[i] >= displs_counter);
            displs_counter += recv_count[i];

            recv_count_int[i] = dal::detail::integral_cast<int>(recv_count[i]);
            displs_int[i] = dal::detail::integral_cast<int>(displs[i]);
        }

        MPI_Request mpi_request;
        try_throw(MPI_Igatherv(send_buf,
                               dal::detail::integral_cast<int>(send_count),
                               MPI_BYTE,
                               recv_buf,
                               recv_count_int.get(),
                               displs_int.get(),
                               MPI_BYTE,
                               dal::detail::integral_cast<int>(root),
                               mpi_comm_,
                               &mpi_request));
        return new mpi_request_impl{ mpi_request };
    }

private:
    static void try_throw(int mpi_error_code) {
        if (mpi_error_code != MPI_SUCCESS) {
            throw mpi_communicator_error{ mpi_error_code };
        }
    }

    MPI_Comm mpi_comm_;
    std::int64_t my_rank_ = -1;
    std::int64_t root_rank_ = 0;
};

class mpi_communicator : public spmd_communicator {
public:
    explicit mpi_communicator(const MPI_Comm& mpi_comm = MPI_COMM_WORLD)
            : spmd_communicator(new mpi_communicator_impl{ mpi_comm }) {}

    void set_root_rank(std::int64_t rank) {
        get_impl<mpi_communicator_impl>().set_root_rank(rank);
    }
};

} // namespace oneapi::dal::detail
