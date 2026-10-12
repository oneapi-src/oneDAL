/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include <condition_variable>
#include <mutex>
#include <thread>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::test {

namespace prof = daal::internal;

static const prof::task_entry& entry_at(std::int64_t idx) {
    return prof::profiler::get_instance()->get_task().kernels.at(idx);
}

// Runs `body` on a new thread and keeps the task it opens alive until `release()` is called,
// so several workers can hold open tasks at the same time.
class held_worker {
public:
    template <typename Body>
    explicit held_worker(Body body)
            : thread_([this, body]() {
                  auto task = body(idx_);
                  {
                      std::lock_guard<std::mutex> lock(m_);
                      opened_ = true;
                  }
                  cv_.notify_all();
                  std::unique_lock<std::mutex> lock(m_);
                  cv_.wait(lock, [this] {
                      return released_;
                  });
              }) {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [this] {
            return opened_;
        });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(m_);
            released_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

    std::int64_t idx() const {
        return idx_;
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    bool opened_ = false;
    bool released_ = false;
    std::int64_t idx_ = -1;
    std::thread thread_;
};

static std::int64_t next_idx() {
    return static_cast<std::int64_t>(prof::profiler::get_instance()->get_task().kernels.size());
}

TEST("profiler attaches concurrent worker tasks to the task that started the region") {
    const std::int64_t outer_idx = next_idx();
    auto outer = prof::profiler::start_task("outer");

    // Worker threads have nothing open on their own stacks, like TBB workers joining a region.
    held_worker first([](std::int64_t& idx) {
        idx = next_idx();
        return prof::profiler::start_task("worker");
    });
    held_worker second([](std::int64_t& idx) {
        idx = next_idx();
        return prof::profiler::start_task("worker");
    });

    REQUIRE(entry_at(first.idx()).parent_idx == outer_idx);
    REQUIRE(entry_at(second.idx()).parent_idx == outer_idx);
    REQUIRE(entry_at(second.idx()).level == entry_at(outer_idx).level + 1);

    first.release();
    second.release();

    // A task opened after the region still nests under `outer`, not under a worker task.
    const std::int64_t after_idx = next_idx();
    {
        auto after = prof::profiler::start_task("after");
        REQUIRE(entry_at(after_idx).parent_idx == outer_idx);
    }
}

TEST("profiler nests a worker's regular task under its own threading task") {
    const std::int64_t outer_idx = next_idx();
    auto outer = prof::profiler::start_task("outer");

    std::int64_t threading_idx = -1;
    std::int64_t inner_idx = -1;
    std::thread worker([&]() {
        threading_idx = next_idx();
        auto threading = prof::profiler::start_threading_task("body");
        inner_idx = next_idx();
        auto inner = prof::profiler::start_task("inner");
    });
    worker.join();

    REQUIRE(entry_at(threading_idx).parent_idx == outer_idx);
    REQUIRE(entry_at(inner_idx).parent_idx == threading_idx);
}

} // namespace oneapi::dal::test
