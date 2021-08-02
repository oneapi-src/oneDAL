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

#include <mutex>
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/communicator.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/thread_communicator.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;

class communicator_test : public te::policy_fixture {
public:
    dal::detail::spmd_communicator create_communicator(std::int64_t rank_count) {
        CAPTURE(rank_count);
        thread_comm_.reset(new te::thread_communicator{ rank_count });
        return dal::detail::spmd_communicator{ *thread_comm_ };
    }

    dal::detail::spmd_communicator create_communicator() {
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
    array<T> array_range(std::int64_t count, T dummy) {
        auto dst = array<T>::empty(count);

        T* dst_ptr = dst.get_mutable_data();
        for (std::int64_t i = 0; i < count; i++) {
            dst_ptr[i] = T(i);
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

    table create_random_table(std::int64_t row_count, std::int64_t column_count) {
        std::uniform_real_distribution<float> uniform{ -10.f, 10.f };

        auto x = array<float>::empty(row_count * column_count);
        float* x_ptr = x.get_mutable_data();

        for (std::int64_t i = 0; i < x.get_count(); i++) {
            x_ptr[i] = uniform(rng_);
        }

        return homogen_table::wrap(x, row_count, column_count);
    }

    std::vector<table> create_random_tables(std::int64_t count) {
        std::uniform_int_distribution<std::int64_t> uniform{ 10, 30 };

        std::vector<table> tables;
        tables.reserve(count);

        for (std::int64_t i = 0; i < count; i++) {
            const std::int64_t row_count = uniform(rng_);
            const std::int64_t column_count = uniform(rng_);
            tables.push_back(create_random_table(row_count, column_count));
        }

        return tables;
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
    std::unique_ptr<te::thread_communicator> thread_comm_;
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

TEST_M(communicator_test, "execute if root rank", "[basic]") {
    auto comm = create_communicator();

    SECTION("default else branch") {
        execute([&](std::int64_t rank) {
            const auto root_val = comm.if_root_rank([]() {
                return std::int64_t(-1);
            });
            exclusive([&]() {
                if (rank == 0) {
                    REQUIRE(root_val == -1);
                }
                else {
                    REQUIRE(root_val == 0);
                }
            });
        });
    }

    SECTION("custom else branch") {
        execute([&](std::int64_t rank) {
            const auto root_val = comm.if_root_rank_else(
                []() {
                    return std::int64_t(-1);
                },
                []() {
                    return std::int64_t(-2);
                });

            exclusive([&]() {
                if (rank == 0) {
                    REQUIRE(root_val == -1);
                }
                else {
                    REQUIRE(root_val == -2);
                }
            });
        });
    }
}

TEST_M(communicator_test, "bcast single value of primitive type", "[bcast]") {
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        constexpr float root_x = 3.14;
        float x = comm.if_root_rank([=] {
            return root_x;
        });

        comm.bcast(x).wait();

        exclusive([&]() {
            REQUIRE(x == root_x);
        });
    });
}

TEST_M(communicator_test, "bcast multiple values of primitive type", "[bcast]") {
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
TEST_M(communicator_test, "USM bcast multiple values of primitive type", "[bcast][usm]") {
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

TEST_M(communicator_test, "bcast array of primitive type", "[bcast]") {
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
TEST_M(communicator_test, "USM bcast array of primitive type", "[bcast][usm]") {
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

TEST_M(communicator_test, "empty bcast is allowed", "[bcast]") {
    auto comm = create_communicator();

    float* empty_buf = nullptr;
    const std::int64_t empty_buf_count = 0;

    execute([&](std::int64_t rank) {
        comm.bcast(empty_buf, empty_buf_count).wait();
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "empty usm bcast is allowed", "[bcast][usm]") {
    auto comm = create_communicator();

    float* empty_buf = nullptr;
    const std::int64_t empty_buf_count = 0;

    execute([&](std::int64_t rank) {
        comm.bcast(this->get_queue(), empty_buf, empty_buf_count).wait();
    });
}
#endif

TEST_M(communicator_test, "gather single value of primitive type", "[gather]") {
    auto comm = create_communicator();

    auto send = this->array_range(comm.get_rank_count(), float());

    execute([&](std::int64_t rank) {
        const auto recv = comm.gather(send[rank]);

        exclusive([&]() {
            if (comm.is_root_rank()) {
                check_if_arrays_equal(recv, send);
            }
            else {
                REQUIRE(recv.get_count() == 0);
            }
        });
    });
}

TEST_M(communicator_test, "gather multiple values of primitive type", "[gather]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    auto send_full = this->array_range(comm.get_rank_count() * count_per_rank, float());

    execute([&](std::int64_t rank) {
        const float* send_buf = send_full.get_data() + rank * count_per_rank;

        array<float> recv;
        float* recv_buf = nullptr;
        if (comm.is_root_rank()) {
            recv = array<float>::empty(comm.get_rank_count() * count_per_rank);
            recv_buf = recv.get_mutable_data();
        }

        comm.gather(send_buf, count_per_rank, recv_buf, count_per_rank).wait();

        exclusive([&]() {
            if (comm.is_root_rank()) {
                check_if_arrays_equal(recv, send_full);
            }
            else {
                REQUIRE(recv.get_count() == 0);
            }
        });
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "usm gather multiple values of primitive type", "[gather][usm]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    auto send_full = this->array_range(comm.get_rank_count() * count_per_rank, float());
    auto send_full_device = this->to_device(send_full);

    execute([&](std::int64_t rank) {
        const float* send_buf = send_full_device.get_data() + rank * count_per_rank;

        array<float> recv;
        float* recv_buf = nullptr;
        if (comm.is_root_rank()) {
            recv.reset(this->get_queue(),
                       comm.get_rank_count() * count_per_rank,
                       sycl::usm::alloc::device);
            recv_buf = recv.get_mutable_data();
        }

        comm.gather(this->get_queue(), send_buf, count_per_rank, recv_buf, count_per_rank).wait();

        exclusive([&]() {
            if (comm.is_root_rank()) {
                check_if_arrays_equal(this->to_host(recv), send_full);
            }
            else {
                REQUIRE(recv.get_count() == 0);
            }
        });
    });
}
#endif

TEST_M(communicator_test, "gather array of primitive type", "[gather]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    auto send_full = this->array_range(comm.get_rank_count() * count_per_rank, float());

    execute([&](std::int64_t rank) {
        const float* send_buf = send_full.get_data() + rank * count_per_rank;
        const auto send = array<float>::wrap(send_buf, count_per_rank);

        array<float> recv;
        if (comm.is_root_rank()) {
            recv.reset(comm.get_rank_count() * count_per_rank);
        }

        comm.gather(send, recv).wait();

        exclusive([&]() {
            if (comm.is_root_rank()) {
                check_if_arrays_equal(recv, send_full);
            }
            else {
                REQUIRE(recv.get_count() == 0);
            }
        });
    });
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(communicator_test, "usm gather array of primitive type", "[gather][usm]") {
    constexpr std::int64_t count_per_rank = 5;
    auto comm = create_communicator();

    auto send_full = this->array_range(comm.get_rank_count() * count_per_rank, float());

    execute([&](std::int64_t rank) {
        const float* send_buf = send_full.get_data() + rank * count_per_rank;
        const auto send = this->to_device(array<float>::wrap(send_buf, count_per_rank));

        array<float> recv;
        if (comm.is_root_rank()) {
            recv.reset(this->get_queue(),
                       comm.get_rank_count() * count_per_rank,
                       sycl::usm::alloc::device);
        }

        comm.gather(send, recv).wait();

        exclusive([&]() {
            if (comm.is_root_rank()) {
                check_if_arrays_equal(this->to_host(recv), send_full);
            }
            else {
                REQUIRE(recv.get_count() == 0);
            }
        });
    });
}
#endif

} // namespace oneapi::dal::test
