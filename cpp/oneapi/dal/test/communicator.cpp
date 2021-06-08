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
#include "oneapi/dal/detail/communicator_utils.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/communicator.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;

class communicator_test {
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
        const bool comm_is_root = dal::detail::is_root_rank(comm);
        exclusive([&]() {
            REQUIRE(comm_is_root == (rank == 0));
        });
    });
}

TEST_M(communicator_test, "execute if root rank", "[basic]") {
    auto comm = create_communicator();

    SECTION("default else branch") {
        execute([&](std::int64_t rank) {
            const auto root_val = dal::detail::if_root_rank(comm, []() {
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
            const auto root_val = dal::detail::if_root_rank_else(
                comm,
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
        float x = dal::detail::if_root_rank(comm, [=] {
            return root_x;
        });

        dal::detail::bcast(comm, x);

        exclusive([&]() {
            REQUIRE(x == root_x);
        });
    });
}

TEST_M(communicator_test, "bcast multiple values of primitive type", "[bcast]") {
    auto comm = create_communicator();

    execute([&](std::int64_t rank) {
        constexpr std::int64_t count = 4;
        std::array<std::int64_t, count> buf = { 0 };

        if (dal::detail::is_root_rank(comm)) {
            for (std::int64_t i = 0; i < count; i++) {
                buf[i] = i;
            }
        }

        dal::detail::bcast(comm, buf.data(), count);

        exclusive([&]() {
            for (std::int64_t i = 0; i < count; i++) {
                REQUIRE(buf[i] == i);
            }
        });
    });
}

TEST_M(communicator_test, "bcast array of primitive type", "[bcast]") {
    auto comm = create_communicator();

    SECTION("array is already allocated") {
        execute([&](std::int64_t rank) {
            constexpr std::int64_t count = 4;
            auto buf = array<float>::empty(count);
            float* buf_ptr = buf.get_mutable_data();

            if (dal::detail::is_root_rank(comm)) {
                for (std::int64_t i = 0; i < count; i++) {
                    buf_ptr[i] = i;
                }
            }

            dal::detail::bcast(comm, buf);

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

    SECTION("array is not allocated") {
        execute([&](std::int64_t rank) {
            constexpr std::int64_t count = 4;
            array<float> buf;

            if (dal::detail::is_root_rank(comm)) {
                buf = array<float>::empty(count);
                float* buf_ptr = buf.get_mutable_data();
                for (std::int64_t i = 0; i < count; i++) {
                    buf_ptr[i] = i;
                }
            }

            dal::detail::bcast(comm, buf);

            exclusive([&]() {
                REQUIRE(buf.get_count() == count);
                REQUIRE(buf.has_mutable_data());

                for (std::int64_t i = 0; i < count; i++) {
                    REQUIRE(buf[i] == i);
                }
            });
        });
    }
}

TEST_M(communicator_test, "bcast single value of serializable type", "[bcast]") {
    auto comm = create_communicator();

    constexpr std::int64_t row_count = 32;
    constexpr std::int64_t column_count = 8;
    const auto origin = create_random_table(row_count, column_count);

    execute([&](std::int64_t rank) {
        table t;
        if (dal::detail::is_root_rank(comm)) {
            t = origin;
        }

        dal::detail::bcast(comm, t);

        exclusive([&]() {
            te::check_if_tables_equal<float>(t, origin);
        });
    });
}

TEST_M(communicator_test, "gather single value of primitive type", "[gather]") {
    auto comm = create_communicator();

    constexpr float magic_x = 3.14;
    std::vector<float> x_per_rank(comm.get_rank_count());
    for (std::int64_t i = 0; i < comm.get_rank_count(); i++) {
        x_per_rank[i] = magic_x * (i + 1);
    }

    execute([&](std::int64_t rank) {
        const auto x_root = dal::detail::gather(comm, x_per_rank[rank]);

        exclusive([&]() {
            if (dal::detail::is_root_rank(comm)) {
                REQUIRE(x_root == x_per_rank);
            }
            else {
                REQUIRE(x_root.size() == 0);
            }
        });
    });
}

TEST_M(communicator_test, "gather multiple values of primitive type", "[gather]") {
    auto comm = create_communicator();

    constexpr float magic_x = 3.14;
    constexpr std::int64_t count_per_rank = 5;

    std::vector<float> x_per_rank(comm.get_rank_count() * count_per_rank);
    for (std::int64_t i = 0; i < std::int64_t(x_per_rank.size()); i++) {
        x_per_rank[i] = magic_x * (i + 1);
    }

    execute([&](std::int64_t rank) {
        auto recv = dal::detail::if_root_rank(comm, [&]() {
            return std::vector<float>(comm.get_rank_count() * count_per_rank);
        });

        const float* send_buf = x_per_rank.data() + rank * count_per_rank;
        float* recv_buf = recv.data();

        dal::detail::gather(comm, send_buf, count_per_rank, recv_buf);

        exclusive([&]() {
            if (dal::detail::is_root_rank(comm)) {
                REQUIRE(recv == x_per_rank);
            }
            else {
                REQUIRE(recv.size() == 0);
            }
        });
    });
}

TEST_M(communicator_test, "gather single value of serializable type", "[gather]") {
    auto comm = create_communicator();

    const auto tables_per_rank = create_random_tables(comm.get_rank_count());

    execute([&](std::int64_t rank) {
        const auto root_tables = dal::detail::gather(comm, tables_per_rank[rank]);

        exclusive([&]() {
            if (dal::detail::is_root_rank(comm)) {
                REQUIRE(root_tables.size() == std::size_t(comm.get_rank_count()));
                for (std::int64_t i = 0; i < comm.get_rank_count(); i++) {
                    te::check_if_tables_equal<float>(root_tables[i], tables_per_rank[i]);
                }
            }
            else {
                REQUIRE(root_tables.size() == 0);
            }
        });
    });
}

} // namespace oneapi::dal::test
