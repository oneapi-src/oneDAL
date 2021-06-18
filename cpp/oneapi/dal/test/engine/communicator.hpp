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

#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <condition_variable>

#include "oneapi/dal/detail/communicator.hpp"
#include "oneapi/dal/detail/communicator_utils.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test::engine {

class thread_communicator_context : public base {
public:
    explicit thread_communicator_context(std::int64_t thread_count) : thread_count_(thread_count) {
        ONEDAL_ASSERT(thread_count > 0);
        thread_pool_.reserve(thread_count);
        thread_id_map_.reserve(thread_count);
    }

    std::int64_t get_root_rank() const {
        return 0;
    }

    std::int64_t get_this_thread_rank() const {
        return map_thread_id_to_rank(std::this_thread::get_id());
    }

    std::int64_t map_thread_id_to_rank(const std::thread::id& thread_id) const {
        return thread_id_map_.at(thread_id);
    }

    std::int64_t get_thread_count() const {
        return thread_count_;
    }

    template <typename Body>
    void execute(const Body& body) {
        // TODO: Use TBB threads
        for (std::int64_t i = 0; i < thread_count_; i++) {
            thread_pool_.emplace_back([=]() {
                init(i);
                body(i);
            });
        }

        for (auto& thread : thread_pool_) {
            thread.join();
        }
    }

    template <typename Body>
    void exclusive(const Body& body) {
        std::unique_lock<std::mutex> lock(mutex_);
        body();
    }

private:
    /// Blocks until all threads are mapped to ranks
    void init(std::int64_t rank) {
        std::unique_lock<std::mutex> lock(mutex_);

        thread_id_map_[std::this_thread::get_id()] = rank;
        if (thread_id_map_.size() == std::size_t(thread_count_)) {
            cv_.notify_all();
        }
        else {
            cv_.wait(lock, [this]() {
                return thread_id_map_.size() == std::size_t(thread_count_);
            });
        }
    }

    std::int64_t thread_count_;
    std::vector<std::thread> thread_pool_;
    std::unordered_map<std::thread::id, std::int64_t> thread_id_map_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

class thread_communicator_barrier {
public:
    explicit thread_communicator_barrier(thread_communicator_context& ctx)
            : ctx_(ctx),
              thread_counter_(ctx.get_thread_count()),
              generation_counter_(0) {}

    template <typename UniqueOp>
    void operator()(UniqueOp&& unique_op) {
        std::unique_lock<std::mutex> lock(mutex_);
        const std::uint64_t thread_generation = generation_counter_;

        if (--thread_counter_ == 0) {
            generation_counter_++;
            thread_counter_ = ctx_.get_thread_count();
            unique_op();
            cv_.notify_all();
        }
        else {
            cv_.wait(lock, [=] {
                return thread_generation != generation_counter_;
            });
        }
    }

    void operator()() {
        (*this)([]() {});
    }

private:
    thread_communicator_context& ctx_;
    std::int64_t thread_counter_;
    std::uint64_t generation_counter_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

class thread_communicator_bcast {
public:
    explicit thread_communicator_bcast(thread_communicator_context& ctx)
            : ctx_(ctx),
              barrier_(ctx),
              source_count_(0),
              source_buf_(nullptr) {}

    void operator()(byte_t* send_buf, std::int64_t count, std::int64_t root) {
        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(count > 0);
        ONEDAL_ASSERT(root >= 0);

        if (ctx_.get_this_thread_rank() == root) {
            source_count_ = count;
            source_buf_ = send_buf;
        }

        barrier_();

        if (ctx_.get_this_thread_rank() != root) {
            ONEDAL_ASSERT(source_buf_);
            ONEDAL_ASSERT(source_count_ > 0);
            ONEDAL_ASSERT(count <= source_count_);

            for (std::int64_t i = 0; i < count; i++) {
                send_buf[i] = source_buf_[i];
            }
        }

        barrier_([&]() {
            source_count_ = 0;
            source_buf_ = nullptr;
        });
    }

private:
    thread_communicator_context& ctx_;
    thread_communicator_barrier barrier_;
    std::int64_t source_count_;
    byte_t* source_buf_;
};

class thread_communicator_gather {
public:
    explicit thread_communicator_gather(thread_communicator_context& ctx)
            : ctx_(ctx),
              barrier_(ctx),
              recv_count_(0),
              recv_buf_(nullptr) {}

    void operator()(const byte_t* send_buf,
                    std::int64_t send_count,
                    byte_t* recv_buf,
                    std::int64_t recv_count,
                    std::int64_t root) {
        ONEDAL_ASSERT(root >= 0);

        if (ctx_.get_this_thread_rank() == root) {
            ONEDAL_ASSERT(recv_buf);
            ONEDAL_ASSERT(recv_count > 0);
            recv_count_ = recv_count;
            recv_buf_ = recv_buf;
        }
        else {
            ONEDAL_ASSERT(send_buf);
            ONEDAL_ASSERT(send_count > 0);
        }

        barrier_();

        ONEDAL_ASSERT(recv_buf_);
        ONEDAL_ASSERT(recv_count_ > 0);
        ONEDAL_ASSERT(send_count <= recv_count_);

        const std::int64_t this_thread_offset = ctx_.get_this_thread_rank() * recv_count_;
        for (std::int64_t i = 0; i < send_count; i++) {
            recv_buf_[this_thread_offset + i] = send_buf[i];
        }

        barrier_([&]() {
            recv_count_ = 0;
            recv_buf_ = nullptr;
        });
    }

private:
    thread_communicator_context& ctx_;
    thread_communicator_barrier barrier_;
    std::int64_t recv_count_;
    byte_t* recv_buf_;
};

class thread_communicator_gatherv {
public:
    explicit thread_communicator_gatherv(thread_communicator_context& ctx)
            : ctx_(ctx),
              barrier_(ctx),
              recv_count_(nullptr),
              displs_(nullptr),
              recv_buf_(nullptr) {}

    void operator()(const byte_t* send_buf,
                    std::int64_t send_count,
                    byte_t* recv_buf,
                    const std::int64_t* recv_count,
                    const std::int64_t* displs,
                    std::int64_t root) {
        ONEDAL_ASSERT(root >= 0);

        const std::int64_t rank = ctx_.get_this_thread_rank();
        if (rank == root) {
            ONEDAL_ASSERT(recv_buf);
            ONEDAL_ASSERT(displs);
            ONEDAL_ASSERT(recv_count);
            recv_count_ = recv_count;
            displs_ = displs;
            recv_buf_ = recv_buf;
        }
        else {
            ONEDAL_ASSERT(send_buf);
            ONEDAL_ASSERT(send_count > 0);
        }

        barrier_();

        ONEDAL_ASSERT(recv_count_);
        ONEDAL_ASSERT(displs_);
        ONEDAL_ASSERT(recv_buf_);
        ONEDAL_ASSERT(send_count <= recv_count_[rank]);

        for (std::int64_t i = 0; i < send_count; i++) {
            recv_buf_[displs_[rank] + i] = send_buf[i];
        }

        barrier_([&]() {
            recv_count_ = nullptr;
            displs_ = nullptr;
            recv_buf_ = nullptr;
        });
    }

private:
    thread_communicator_context& ctx_;
    thread_communicator_barrier barrier_;
    const std::int64_t* recv_count_;
    const std::int64_t* displs_;
    byte_t* recv_buf_;
};

class thread_communicator_request_impl : public dal::detail::spmd_request_iface {
public:
    thread_communicator_request_impl() = default;

    void wait() override {}

    bool test() override {
        return true;
    }
};

class thread_communicator_impl : public dal::detail::spmd_communicator_iface {
public:
    explicit thread_communicator_impl(std::int64_t thread_count)
            : ctx_(thread_count),
              barrier_(ctx_),
              bcast_(ctx_),
              gather_(ctx_),
              gatherv_(ctx_) {}

    std::int64_t get_rank() override {
        return ctx_.map_thread_id_to_rank(std::this_thread::get_id());
    }

    std::int64_t get_root_rank() override {
        return ctx_.get_root_rank();
    }

    std::int64_t get_rank_count() override {
        return ctx_.get_thread_count();
    }

    void barrier() override {
        barrier_();
    }

    dal::detail::spmd_request_iface* bcast(byte_t* send_buf,
                                           std::int64_t count,
                                           std::int64_t root) override {
        bcast_(send_buf, count, root);
        return new thread_communicator_request_impl{};
    }

    dal::detail::spmd_request_iface* gather(const byte_t* send_buf,
                                            std::int64_t send_count,
                                            byte_t* recv_buf,
                                            std::int64_t recv_count,
                                            std::int64_t root) override {
        gather_(send_buf, send_count, recv_buf, recv_count, root);
        return new thread_communicator_request_impl{};
    }

    dal::detail::spmd_request_iface* gatherv(const byte_t* send_buf,
                                             std::int64_t send_count,
                                             byte_t* recv_buf,
                                             const std::int64_t* recv_count,
                                             const std::int64_t* displs,
                                             std::int64_t root) override {
        gatherv_(send_buf, send_count, recv_buf, recv_count, displs, root);
        return new thread_communicator_request_impl{};
    }

    thread_communicator_context& get_context() {
        return ctx_;
    }

private:
    thread_communicator_context ctx_;
    thread_communicator_barrier barrier_;
    thread_communicator_bcast bcast_;
    thread_communicator_gather gather_;
    thread_communicator_gatherv gatherv_;
};

class thread_communicator : public dal::detail::spmd_communicator {
public:
    explicit thread_communicator(std::int64_t thread_count)
            : dal::detail::spmd_communicator(new thread_communicator_impl{ thread_count }) {}

    std::int64_t get_root_rank() const {
        return get_impl<thread_communicator_impl>().get_root_rank();
    }

    template <typename Body>
    void execute(const Body& body) {
        get_impl<thread_communicator_impl>().get_context().execute(body);
    }

    template <typename Body>
    void exclusive(const Body& body) {
        get_impl<thread_communicator_impl>().get_context().exclusive(body);
    }

    template <typename Body>
    auto map(const Body& body) {
        using map_t = decltype(body(std::declval<std::int64_t>()));

        std::vector<map_t> results(get_rank_count());
        execute([&](std::int64_t rank) {
            results[rank] = body(rank);
        });
        return results;
    }
};

} // namespace oneapi::dal::test::engine
