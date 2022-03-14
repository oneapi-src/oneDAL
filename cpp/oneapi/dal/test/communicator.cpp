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

#include <mutex>
#include "oneapi/dal/spmd/detail/communicator_utils.hpp"
#include "oneapi/dal/detail/communicator.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/thread_communicator.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::test {

namespace de = dal::detail;
namespace te = dal::test::engine;

class communicator_test : public te::policy_fixture {
public:
#ifdef ONEDAL_DATA_PARALLEL
    using comm_t = te::thread_communicator<spmd::device_memory_access::usm>;
    using spmd_comm_t = spmd::communicator<spmd::device_memory_access::usm>;
#else
    using comm_t = te::thread_communicator<spmd::device_memory_access::none>;
    using spmd_comm_t = spmd::communicator<spmd::device_memory_access::none>;
#endif
    auto create_communicator(std::int64_t rank_count) {
        CAPTURE(rank_count);
#ifdef ONEDAL_DATA_PARALLEL
        thread_comm_.reset(new comm_t{ this->get_queue(), rank_count });
#else
        thread_comm_.reset(new comm_t{ rank_count });
#endif
        return spmd_comm_t{ *thread_comm_ };
    }

    auto create_communicator() {
        return create_communicator(generate_thread_count());
    }

    std::int64_t generate_thread_count() const {
        return GENERATE(1, 2, 4, 8);
    }

    template <typename Body>
    void execute(Body&& body) {
        if (!thread_comm_) {
            create_communicator(generate_thread_count());
        }
        thread_comm_->execute(body);
    }

    template <typename Body>
    void exclusive(Body&& body) {
        std::scoped_lock<std::mutex> lock{ mtx_ };
        body();
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    array<T> to_device(const array<T>& src) {
        auto dst = array<T>::empty(this->get_queue(), src.get_count(), sycl::usm::alloc::device);
        dal::detail::memcpy_host2usm(this->get_queue(),
                                     dst.get_mutable_data(),
                                     src.get_data(),
                                     src.get_size());
        return dst;
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    array<T> to_host(const array<T>& src) {
        auto dst = array<T>::empty(src.get_count());
        dal::detail::memcpy_usm2host(this->get_queue(),
                                     dst.get_mutable_data(),
                                     src.get_data(),
                                     src.get_size());
        return dst;
    }
#endif

    template <typename T>
    array<T> array_range(std::int64_t count, T start = T(0)) {
        auto dst = array<T>::empty(count);

        T* dst_ptr = dst.get_mutable_data();
        for (std::int64_t i = 0; i < count; i++) {
            dst_ptr[i] = start + T(i);
        }

        return dst;
    }

    template <typename T>
    array<T> array_full(std::int64_t count, T value) {
        auto dst = array<T>::empty(count);

        T* dst_ptr = dst.get_mutable_data();
        for (std::int64_t i = 0; i < count; i++) {
            dst_ptr[i] = value;
        }

        return dst;
    }

    template <typename T>
    array<T> array_displacements(const array<T>& values) {
        const auto displs = array<T>::empty(values.get_count());
        T* displs_ptr = displs.get_mutable_data();

        T sum = T(0);
        for (std::int64_t i = 0; i < displs.get_count(); i++) {
            displs_ptr[i] = sum;
            sum += values[i];
        }

        return displs;
    }

    template <typename T>
    T array_sum(const array<T>& values) {
        T sum = T(0);
        for (std::int64_t i = 0; i < values.get_count(); i++) {
            sum += values[i];
        }
        return sum;
    }

    template <typename T>
    void check_if_arrays_equal(const array<T>& actual, const array<T>& expected) {
        REQUIRE(actual.get_count() == expected.get_count());
        for (std::int64_t i = 0; i < actual.get_count(); i++) {
            CAPTURE(i);
            REQUIRE(actual[i] == expected[i]);
        }
    }

private:
    static constexpr int seed = 7777;

    std::mt19937 rng_{ seed };
    std::mutex mtx_;
    std::unique_ptr<comm_t> thread_comm_;
};

TEST_M(communicator_test, "get current rank", "[basic]") {
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        const std::int64_t comm_rank = comm.get_rank();
        exclusive([&]() {
            REQUIRE(comm_rank == rank);
        });
    });
}

TEST_M(communicator_test, "check if root rank", "[basic]") {
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        const bool comm_is_root = comm.is_root_rank();
        exclusive([&]() {
            REQUIRE(comm_is_root == (rank == 0));
        });
    });
}

TEST_M(communicator_test, "bcast single value", "[bcast]") {
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        constexpr float root_x = 3.14;
        float x = de::if_root_rank(comm, [=] {
            return root_x;
        });

        comm.bcast(x).wait();

        exclusive([&]() {
            REQUIRE(x == root_x);
        });
    });
}

TEST_M(communicator_test, "bcast multiple values", "[bcast]") {
    constexpr std::int64_t count = 4;
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        array<std::int64_t> buf;

        if (comm.is_root_rank()) {
            buf = this->array_range(count, std::int64_t());
        }
        else {
            buf = this->array_full(count, std::int64_t(0));
        }

        comm.bcast(buf.get_mutable_data(), count).wait();

        exclusive([&]() {
            for (std::int64_t i = 0; i < count; i++) {
                REQUIRE(buf[i] == i);
            }
        });
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "USM bcast multiple values", "[bcast][usm]") {
    constexpr std::int64_t count = 4;
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        array<std::int64_t> buf;

        if (comm.is_root_rank()) {
            buf = this->to_device(this->array_range(count, std::int64_t()));
        }
        else {
            buf = this->to_device(this->array_full(count, std::int64_t(0)));
        }

        comm.bcast(this->get_queue(), buf.get_mutable_data(), count).wait();

        exclusive([&]() {
            const auto buf_host = this->to_host(buf);
            for (std::int64_t i = 0; i < count; i++) {
                REQUIRE(buf_host[i] == i);
            }
        });
    });
}
#endif

TEST_M(communicator_test, "bcast array", "[bcast][array]") {
    constexpr std::int64_t count = 4;
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        array<std::int64_t> buf;

        if (comm.is_root_rank()) {
            buf = this->array_range(count, std::int64_t());
        }
        else {
            buf = this->array_full(count, std::int64_t(0));
        }
        const std::int64_t* buf_ptr = buf.get_mutable_data();

        comm.bcast(buf).wait();

        exclusive([&]() {
            // Make sure we do not reallocate memory
            REQUIRE(buf.get_data() == buf_ptr);
            REQUIRE(buf.get_mutable_data() == buf_ptr);
            REQUIRE(buf.get_count() == count);

            for (std::int64_t i = 0; i < count; i++) {
                REQUIRE(buf[i] == i);
            }
        });
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "USM bcast array", "[bcast][usm][array]") {
    constexpr std::int64_t count = 4;
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        array<std::int64_t> buf;

        if (comm.is_root_rank()) {
            buf = this->to_device(this->array_range(count, std::int64_t()));
        }
        else {
            buf = this->to_device(this->array_full(count, std::int64_t(0)));
        }
        const std::int64_t* buf_ptr = buf.get_mutable_data();

        comm.bcast(buf).wait();

        exclusive([&]() {
            // Make sure we do not reallocate memory
            REQUIRE(buf.get_data() == buf_ptr);
            REQUIRE(buf.get_mutable_data() == buf_ptr);
            REQUIRE(buf.get_count() == count);

            const auto buf_host = this->to_host(buf);
            for (std::int64_t i = 0; i < count; i++) {
                REQUIRE(buf_host[i] == i);
            }
        });
    });
}
#endif

TEST_M(communicator_test, "empty bcast is allowed", "[bcast][empty]") {
    auto comm = create_communicator();

    float* empty_buf = nullptr;
    const std::int64_t empty_buf_count = 0;
    execute([&](std::int64_t rank) {
        comm.bcast(empty_buf, empty_buf_count).wait();
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "empty USM bcast is allowed", "[bcast][usm][empty]") {
    auto comm = create_communicator();

    float* empty_buf = nullptr;
    const std::int64_t empty_buf_count = 0;

    execute([&](std::int64_t rank) {
        comm.bcast(this->get_queue(), empty_buf, empty_buf_count).wait();
    });
}
#endif
TEST_M(communicator_test, "allgather array", "[allgather]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    auto send_full = this->array_range(comm.get_rank_count() * count_per_rank, float());

    execute([&](std::int64_t rank) {
        const float* send_buf = send_full.get_data() + rank * count_per_rank;
        const auto send = array<float>::empty(count_per_rank);
        for (std::int64_t i = 0; i < count_per_rank; i++)
            send.get_mutable_data()[i] = send_buf[i];
        const auto recv = array<float>::empty(comm.get_rank_count() * count_per_rank);

        comm.allgather(send, recv).wait();

        exclusive([&]() {
            check_if_arrays_equal(recv, send_full);
        });
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "USM allgather array", "[allgather][usm]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    auto send_full = this->array_range(comm.get_rank_count() * count_per_rank, float());
    auto send_full_device = this->to_device(send_full);

    execute([&](std::int64_t rank) {
        const float* send_buf = send_full_device.get_data() + rank * count_per_rank;
        const auto send = array<float>::wrap(this->get_queue(), send_buf, count_per_rank);
        const auto recv = array<float>::empty(this->get_queue(),
                                              comm.get_rank_count() * count_per_rank,
                                              sycl::usm::alloc::device);

        comm.allgather(send, recv).wait();

        exclusive([&]() {
            check_if_arrays_equal(this->to_host(recv), send_full);
        });
    });
}
#endif

TEST_M(communicator_test, "allreduce single value", "[allreduce]") {
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        float x = 1;

        comm.allreduce(x, spmd::reduce_op::sum).wait();

        exclusive([&]() {
            REQUIRE(std::int64_t(x) == comm.get_rank_count());
        });
    });
}

TEST_M(communicator_test, "allreduce multiple values", "[allreduce]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        const auto send = this->array_full(count_per_rank, float(1));
        const auto recv = this->array_full(count_per_rank, float(0));

        comm.allreduce(send.get_data(),
                       recv.get_mutable_data(),
                       count_per_rank,
                       spmd::reduce_op::sum)
            .wait();

        exclusive([&]() {
            const auto expected = this->array_full(count_per_rank, float(comm.get_rank_count()));
            check_if_arrays_equal(recv, expected);
        });
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "USM allreduce multiple values", "[allreduce][usm]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        const auto send = this->to_device(this->array_full(count_per_rank, float(1)));
        const auto recv = this->to_device(this->array_full(count_per_rank, float(0)));

        comm.allreduce(this->get_queue(),
                       send.get_data(),
                       recv.get_mutable_data(),
                       count_per_rank,
                       spmd::reduce_op::sum)
            .wait();

        exclusive([&]() {
            const auto expected = this->array_full(count_per_rank, float(comm.get_rank_count()));
            check_if_arrays_equal(this->to_host(recv), expected);
        });
    });
}
#endif

TEST_M(communicator_test, "allreduce array", "[allreduce]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        const auto ary = this->array_full(count_per_rank, float(1));

        comm.allreduce(ary, spmd::reduce_op::sum).wait();

        exclusive([&]() {
            const auto expected = this->array_full(count_per_rank, float(comm.get_rank_count()));
            check_if_arrays_equal(ary, expected);
        });
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "USM allreduce array", "[allreduce][usm]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        const auto ary = this->to_device(this->array_full(count_per_rank, float(1)));

        comm.allreduce(ary, spmd::reduce_op::sum).wait();

        exclusive([&]() {
            const auto expected = this->array_full(count_per_rank, float(comm.get_rank_count()));
            check_if_arrays_equal(this->to_host(ary), expected);
        });
    });
}
#endif

TEST_M(communicator_test, "empty allreduce is allowed", "[allreduce][empty]") {
    auto comm = create_communicator();
    execute([&](std::int64_t rank) {
        const float* send_buf = nullptr;
        float* recv_buf = nullptr;
        comm.allreduce(send_buf, recv_buf, 0, spmd::reduce_op::sum).wait();
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "empty USM allreduce is allowed", "[allreduce][usm][empty]") {
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        const byte_t* send_buf = nullptr;
        byte_t* recv_buf = nullptr;
        comm.allreduce(this->get_queue(), send_buf, recv_buf, 0, spmd::reduce_op::sum).wait();
    });
}
#endif

TEST_M(communicator_test, "sendrecv_replace", "[sendrecv_replace][single value]") {
    constexpr std::int64_t count_per_rank = 1;
    auto comm = create_communicator();
    execute([&](std::int64_t rank) {
        const auto rank_count = comm.get_rank_count();
        const std::int64_t source_rank = rank == rank_count - 1 ? 0 : rank + 1;
        const std::int64_t destination_rank = rank == 0 ? rank_count - 1 : rank - 1;
        auto ary = this->array_full(count_per_rank, float(rank));
        comm.sendrecv_replace(ary, destination_rank, source_rank).wait();
        exclusive([&]() {
            const auto expected = this->array_full(count_per_rank, float(source_rank));
            check_if_arrays_equal(ary, expected);
        });
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "sendrecv_replace USM", "[sendrecv_replace][usm][single value]") {
    constexpr std::int64_t count_per_rank = 1;
    auto comm = create_communicator();
    execute([&](std::int64_t rank) {
        const auto rank_count = comm.get_rank_count();
        const std::int64_t source_rank = rank == rank_count - 1 ? 0 : rank + 1;
        const std::int64_t destination_rank = rank == 0 ? rank_count - 1 : rank - 1;
        auto ary = this->to_device(this->array_full(count_per_rank, float(rank)));
        comm.sendrecv_replace(ary, destination_rank, source_rank).wait();
        exclusive([&]() {
            const auto expected = this->array_full(count_per_rank, float(source_rank));
            check_if_arrays_equal(this->to_host(ary), expected);
        });
    });
}
#endif

} // namespace oneapi::dal::test
