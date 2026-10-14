/* file: service_profiler.cpp */
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

/*
//++
//  Recording, merging and reporting for the kernel profiler.
//
//  Each thread records into its own tree of call paths, so opening and closing a task takes no lock
//  and memory grows with the number of distinct paths, not with the number of runs. The threading
//  layer tells each thread which task started the parallel region of the chunk it runs, and the
//  trees are joined there when the report is built.
//--
*/

#include "src/services/service_profiler.h"
#include "src/threading/threading.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace daal
{
namespace internal
{
namespace
{

constexpr int memory_kinds   = 2;
constexpr int transfer_kinds = 2;

inline std::uint64_t now_ns()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

inline void atomic_max(std::atomic<std::int64_t> & target, std::int64_t value)
{
    std::int64_t current = target.load(std::memory_order_relaxed);
    while (current < value && !target.compare_exchange_weak(current, value, std::memory_order_relaxed))
    {
    }
}

std::string format_time(std::uint64_t time_ns)
{
    std::ostringstream out;
    const double time = static_cast<double>(time_ns);
    if (time_ns == 0)
        out << "0 ns";
    else if (time >= 1e9)
        out << std::fixed << std::setprecision(2) << time / 1e9 << " s";
    else if (time >= 1e6)
        out << std::fixed << std::setprecision(2) << time / 1e6 << " ms";
    else if (time >= 1e3)
        out << std::fixed << std::setprecision(2) << time / 1e3 << " us";
    else
        out << time_ns << " ns";
    return out.str();
}

inline std::uint64_t positive_part(std::int64_t value)
{
    return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

/// Allocations, deallocations and copies charged to one task, not including its children.
struct own_stats
{
    std::uint64_t alloc_bytes[memory_kinds]      = {};
    std::uint64_t alloc_count[memory_kinds]      = {};
    std::uint64_t free_bytes[memory_kinds]       = {};
    std::uint64_t free_count[memory_kinds]       = {};
    std::uint64_t transfer_bytes[transfer_kinds] = {};
    std::uint64_t transfer_count[transfer_kinds] = {};

    void add(const own_stats & other)
    {
        for (int k = 0; k < memory_kinds; ++k)
        {
            alloc_bytes[k] += other.alloc_bytes[k];
            alloc_count[k] += other.alloc_count[k];
            free_bytes[k] += other.free_bytes[k];
            free_count[k] += other.free_count[k];
        }
        for (int k = 0; k < transfer_kinds; ++k)
        {
            transfer_bytes[k] += other.transfer_bytes[k];
            transfer_count[k] += other.transfer_count[k];
        }
    }
};

/// A call path in one thread's tree. Only the owning thread writes to it.
struct node
{
    std::string name;
    const char * key = nullptr; // the pointer first seen for this name, compared before the string
    node * parent    = nullptr; // nullptr for a top-level task of the thread
    node * anchor    = nullptr; // for a top-level task of a worker: the task that started the region
    bool threading   = false;
    bool in_parallel = false; // ran inside a parallel region at least once
    std::vector<node *> children;

    std::uint64_t count              = 0;
    std::uint64_t total              = 0;
    std::uint64_t min                = UINT64_MAX;
    std::uint64_t max                = 0;
    std::uint64_t peak[memory_kinds] = {};
    std::uint64_t self_total         = 0; // time not in child tasks or in regions this task waited for
    std::uint64_t sequential_total   = 0; // time of the runs outside parallel regions
    std::uint64_t wait_total         = 0; // time waiting for parallel regions beyond running its chunks
    std::uint64_t wait_count         = 0;
    std::uint64_t wait_max           = 0;
    std::uint64_t wait_min           = UINT64_MAX;
    own_stats own;

    // Run of the innermost sequential task, published for the workers that attach to this task.
    std::atomic<std::uint64_t> current_run { 0 };

    // For a task in a parallel region: this thread's time per run of the region, so the report can
    // take the busiest thread of each run rather than of the whole program.
    std::vector<std::pair<std::uint64_t, std::uint64_t> > runs;
    std::uint64_t unsplit_total = 0; // time recorded after `runs` reached max_runs_per_node
};

constexpr std::size_t max_runs_per_node = 1 << 16;
constexpr std::size_t max_regions       = 1 << 22;

struct frame
{
    node * task                              = nullptr;
    std::uint64_t start                      = 0;
    std::uint64_t run                        = 0;     // the region run, or the sequential run, this task is part of
    std::uint64_t paused                     = 0;     // time this thread spent on chunks of other regions while this task waited
    std::uint64_t child_time                 = 0;     // time in child tasks and in parallel regions this task started
    bool counts_in_parent                    = false; // false for the first task of a chunk that interrupted other work
    bool in_parallel                         = false;
    bool sequential                          = false; // sequential tasks track memory process-wide, others per thread
    node * previous_anchor                   = nullptr;
    std::int64_t live_at_start[memory_kinds] = {};
    std::int64_t peak_live[memory_kinds]     = {}; // per-thread tasks only
    std::int64_t outer_peak[memory_kinds]    = {}; // sequential tasks only: the enclosing task's peak so far
};

struct trace_event
{
    const node * task;
    std::uint64_t start;
    std::uint64_t end;
    bool parallel;
};

struct memory_sample
{
    std::uint64_t time;
    std::int64_t live;
    int kind;
};

/// A chunk of a parallel region that a thread is running.
struct chunk
{
    node * parent          = nullptr; // the task the region was started from, or nullptr
    std::uint64_t run      = 0;
    std::size_t base       = 0;     // stack depth when the chunk started
    std::size_t pause_from = 0;     // first stack frame the chunk interrupted
    bool interrupts        = false; // the thread was busy with other work when it took the chunk
    std::uint64_t start    = 0;
};

struct thread_state
{
    std::size_t index = 0;
    node root;
    std::deque<node> nodes; // a deque keeps node addresses stable
    std::vector<frame> stack;
    std::vector<chunk> chunks; // innermost last

    // Blocking regions this thread started and is waiting for, innermost last.
    struct started_region
    {
        std::uint64_t run;
        std::uint64_t start;
        std::size_t depth;        // stack depth of the task that started it
        std::uint64_t child_time; // that task's child time when the region started
    };
    std::vector<started_region> started;
    std::int64_t live[memory_kinds] = {};
    std::unordered_map<const node *, own_stats> anchored; // charged while idle inside a region
    own_stats outside;                                    // charged with no task open anywhere
    std::vector<trace_event> events;
    std::vector<memory_sample> samples;
};

struct allocation_shard
{
    std::mutex mutex;
    std::unordered_map<const void *, std::pair<std::size_t, int> > blocks;
};

constexpr std::size_t shard_count = 64;

struct global_state
{
    std::mutex threads_mutex;
    std::vector<std::unique_ptr<thread_state> > threads;

    // Innermost sequential task open anywhere; a thread with nothing open attaches to it.
    std::atomic<node *> anchor { nullptr };
    std::atomic<std::uint64_t> started_runs { 0 };

    // For each parallel region: the run it was started in and the thread that started it.
    std::mutex regions_mutex;
    std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::size_t> > regions;

    allocation_shard shards[shard_count];
    std::atomic<std::int64_t> live[memory_kinds];
    std::atomic<std::int64_t> region_peak[memory_kinds]; // peak since the innermost sequential task started
    std::atomic<std::int64_t> process_peak[memory_kinds];
    std::atomic<std::uint64_t> untracked_frees { 0 };

    std::atomic<std::int64_t> trace_budget { 0 };
    std::atomic<std::uint64_t> trace_dropped { 0 };
    std::uint64_t origin = now_ns();

    global_state()
    {
        for (int k = 0; k < memory_kinds; ++k)
        {
            live[k]         = 0;
            region_peak[k]  = 0;
            process_peak[k] = 0;
        }
        trace_budget = trace_limit();
    }

    static std::int64_t trace_limit()
    {
        const char * value = std::getenv("ONEDAL_PROFILER_TRACE_LIMIT");
        if (value)
        {
            char * end        = nullptr;
            const long long n = std::strtoll(value, &end, 10);
            if (end != value && *end == '\0' && n >= 0) return n;
        }
        return 4000000;
    }
};

/// Where the logger, the tracer and the report go. Never destroyed, like global_state.
struct log_sink
{
    std::mutex mutex;
    std::ofstream file;
    bool to_file    = false;
    bool every_call = false; // otherwise each call site is logged once
    bool noted      = false;
    std::unordered_set<std::string> seen;

    log_sink()
    {
        const char * path = std::getenv("ONEDAL_PROFILER_OUTPUT");
        if (path && *path)
        {
            file.open(path, std::ios::out | std::ios::trunc);
            to_file = file.is_open();
            if (!to_file) std::cerr << "ONEDAL PROFILER: cannot open " << path << ", writing to stderr" << std::endl;
        }
        // A pipe that nobody reads while oneDAL runs (a notebook kernel) blocks the process once it is
        // full, so stderr gets one line per call site. A file takes every call.
        every_call = to_file;
    }

    std::ostream & out() { return to_file ? static_cast<std::ostream &>(file) : std::cerr; }
};

log_sink & sink()
{
    static log_sink * instance = new log_sink();
    return *instance;
}

void report_at_exit();
profiler_region region_capture(bool blocking);
void region_finish(const profiler_region & region);
void region_enter(const profiler_region & region);
void region_leave();

const profiler_region_hooks region_hooks = { region_capture, region_finish, region_enter, region_leave };

// Never destroyed: tasks can still close from static destructors that run after the report.
global_state & state()
{
    static global_state * instance = [] {
        global_state * s = new global_state();
        if (is_analyzer_enabled() || profiler_trace_path()) std::atexit(report_at_exit);
        // Regions started from now on report their chunks. Not done at load time: on Windows that
        // would load the threading library from inside DllMain.
        if (is_profiler_enabled()) _daal_threader_set_profiler_hooks(&region_hooks);
        return s;
    }();
    return *instance;
}

thread_local thread_state * current_thread_state = nullptr;

thread_state & this_thread()
{
    if (!current_thread_state)
    {
        auto created  = std::make_unique<thread_state>();
        auto & global = state();
        std::lock_guard<std::mutex> lock(global.threads_mutex);
        created->index       = global.threads.size();
        current_thread_state = created.get();
        global.threads.push_back(std::move(created));
    }
    return *current_thread_state;
}

allocation_shard & shard_for(const void * ptr)
{
    const auto bits = reinterpret_cast<std::uintptr_t>(ptr) >> 4;
    return state().shards[(bits * 0x9E3779B97F4A7C15ull) >> 58];
}

/// @param[in] stable_name  Whether `name` outlives the profiler, as the string literals in the macros
///                         do; only then is its address kept for a faster lookup.
node * find_child(thread_state & ts, node * parent, node * anchor, const char * name, bool threading, bool stable_name = true)
{
    auto & children = parent ? parent->children : ts.root.children;
    for (node * child : children)
    {
        if (child->key == name && child->anchor == anchor) return child;
    }
    for (node * child : children)
    {
        if (child->anchor == anchor && child->name == name) return child;
    }
    ts.nodes.emplace_back();
    node * created     = &ts.nodes.back();
    created->name      = name;
    created->key       = stable_name ? name : nullptr;
    created->parent    = parent;
    created->anchor    = anchor;
    created->threading = threading;
    children.push_back(created);
    return created;
}

/// @param[in] shared   Whether the task may publish itself as the anchor and track memory process-wide.
///                     False only for the private calibration state.
constexpr std::size_t max_args_variants = 8;

/// The row name for a task with arguments: "name [args]", or "name [other args]" once the task has
/// max_args_variants rows under this parent already.
std::string args_label(const std::vector<node *> & siblings, const node * anchor, const char * name, const std::string & args)
{
    std::string trimmed = args;
    while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == ';')) trimmed.pop_back();
    const std::string prefix = std::string(name) + " [";
    std::string label        = prefix + trimmed + "]";
    std::size_t variants     = 0;
    for (const node * sibling : siblings)
    {
        if (sibling->anchor != anchor || sibling->name.compare(0, prefix.size(), prefix) != 0) continue;
        if (sibling->name == label) return label;
        ++variants;
    }
    return variants < max_args_variants ? label : prefix + "other args]";
}

/// @param[in] args     The task's arguments, or nullptr
int begin_task(thread_state & ts, const char * name, bool threading, bool shared, const std::string * args = nullptr)
{
    auto & global = state();
    frame f;
    node * parent    = nullptr;
    node * anchor    = nullptr;
    bool in_parallel = true;

    const chunk * c = ts.chunks.empty() ? nullptr : &ts.chunks.back();
    if (c && ts.stack.size() == c->base)
    {
        // The first task of a chunk belongs under the task that started the region, whatever else
        // this thread has open.
        if (!ts.stack.empty() && ts.stack.back().task == c->parent)
            parent = c->parent;
        else
            anchor = c->parent;
        f.run = c->run;
    }
    else if (!ts.stack.empty())
    {
        parent      = ts.stack.back().task;
        in_parallel = threading || ts.stack.back().in_parallel;
        f.run       = ts.stack.back().run;
    }
    else
    {
        // Not inside a chunk the threading layer reported: attach to the sequential task open elsewhere.
        anchor      = shared ? global.anchor.load(std::memory_order_acquire) : nullptr;
        in_parallel = threading || anchor != nullptr;
        f.run       = anchor ? anchor->current_run.load(std::memory_order_relaxed) : 0;
    }
    if (args)
    {
        const std::string label = args_label(parent ? parent->children : ts.root.children, anchor, name, *args);
        f.task                  = find_child(ts, parent, anchor, label.c_str(), threading, false);
    }
    else
    {
        f.task = find_child(ts, parent, anchor, name, threading);
    }
    f.task->in_parallel |= in_parallel;
    f.in_parallel      = in_parallel;
    f.counts_in_parent = parent != nullptr;

    f.sequential = !in_parallel && shared;
    if (f.sequential)
    {
        f.run = global.started_runs.fetch_add(1, std::memory_order_relaxed) + 1;
        f.task->current_run.store(f.run, std::memory_order_relaxed);
        f.previous_anchor = global.anchor.exchange(f.task, std::memory_order_acq_rel);
        for (int k = 0; k < memory_kinds; ++k)
        {
            f.live_at_start[k] = global.live[k].load(std::memory_order_relaxed);
            f.outer_peak[k]    = global.region_peak[k].exchange(f.live_at_start[k], std::memory_order_relaxed);
        }
    }
    else
    {
        for (int k = 0; k < memory_kinds; ++k)
        {
            f.live_at_start[k] = ts.live[k];
            f.peak_live[k]     = ts.live[k];
        }
    }
    ts.stack.push_back(f);
    // Taken last, so the bookkeeping above is charged to the parent rather than to this task.
    ts.stack.back().start = now_ns();
    return static_cast<int>(ts.stack.size() - 1);
}

void end_one(thread_state & ts, std::uint64_t end, bool shared)
{
    auto & global = state();
    const frame f = ts.stack.back();
    ts.stack.pop_back();

    node & task                 = *f.task;
    const std::uint64_t elapsed = end > f.start ? end - f.start : 0;
    const std::uint64_t length  = elapsed > f.paused ? elapsed - f.paused : 0;
    task.count += 1;
    task.total += length;
    task.self_total += length > f.child_time ? length - f.child_time : 0;
    if (f.counts_in_parent && !ts.stack.empty()) ts.stack.back().child_time += length;
    task.min = std::min(task.min, length);
    task.max = std::max(task.max, length);

    if (f.sequential)
    {
        for (int k = 0; k < memory_kinds; ++k)
        {
            const std::int64_t peak = global.region_peak[k].load(std::memory_order_relaxed);
            task.peak[k]            = std::max(task.peak[k], positive_part(peak - f.live_at_start[k]));
            // The enclosing task's peak is the larger of its own so far and this task's.
            atomic_max(global.region_peak[k], f.outer_peak[k]);
        }
        global.anchor.store(f.previous_anchor, std::memory_order_release);
    }
    if (!f.in_parallel)
    {
        task.sequential_total += length;
    }
    else
    {
        auto & runs = task.runs;
        if (!runs.empty() && runs.back().first == f.run)
            runs.back().second += length;
        else if (runs.size() < max_runs_per_node)
            runs.emplace_back(f.run, length);
        else
            task.unsplit_total += length;
        for (int k = 0; k < memory_kinds; ++k)
        {
            task.peak[k] = std::max(task.peak[k], positive_part(f.peak_live[k] - f.live_at_start[k]));
            if (!ts.stack.empty() && !ts.stack.back().sequential)
            {
                ts.stack.back().peak_live[k] = std::max(ts.stack.back().peak_live[k], f.peak_live[k]);
            }
        }
    }

    if (shared && profiler_trace_path())
    {
        if (global.trace_budget.fetch_sub(1, std::memory_order_relaxed) > 0)
            ts.events.push_back({ &task, f.start, end, f.in_parallel });
        else
            global.trace_dropped.fetch_add(1, std::memory_order_relaxed);
    }

    if (shared && is_tracer_enabled())
    {
        const std::string key = "tracer " + task.name;
        if (!task.threading)
            profiler_log(key.c_str(), task.name + " " + format_time(length) + "\n");
        else
            profiler_log(key.c_str(), "THREADING " + task.name
                                          + " finished on the main rank(time could be different for other ranks): " + format_time(length) + "\n");
    }
}

void end_task(thread_state & ts, int depth, bool shared)
{
    const std::uint64_t end = now_ns();
    if (depth < 0) return;
    while (ts.stack.size() > static_cast<std::size_t>(depth)) end_one(ts, end, shared);
}

/// The stats that an allocation or copy on this thread is charged to right now.
own_stats & charge_target(thread_state & ts)
{
    if (!ts.chunks.empty() && ts.stack.size() == ts.chunks.back().base)
    {
        const node * parent = ts.chunks.back().parent;
        return parent ? ts.anchored[parent] : ts.outside;
    }
    if (!ts.stack.empty()) return ts.stack.back().task->own;
    const node * anchor = state().anchor.load(std::memory_order_acquire);
    if (anchor) return ts.anchored[anchor];
    return ts.outside;
}

void record_sample(thread_state & ts, int kind, std::int64_t live)
{
    auto & global = state();
    if (global.trace_budget.fetch_sub(1, std::memory_order_relaxed) > 0)
        ts.samples.push_back({ now_ns(), live, kind });
    else
        global.trace_dropped.fetch_add(1, std::memory_order_relaxed);
}

void update_live(thread_state & ts, int kind, std::int64_t delta)
{
    auto & global            = state();
    const std::int64_t value = global.live[kind].fetch_add(delta, std::memory_order_relaxed) + delta;
    if (delta > 0)
    {
        atomic_max(global.region_peak[kind], value);
        atomic_max(global.process_peak[kind], value);
    }
    ts.live[kind] += delta;
    if (delta > 0 && !ts.stack.empty() && !ts.stack.back().sequential)
    {
        ts.stack.back().peak_live[kind] = std::max(ts.stack.back().peak_live[kind], ts.live[kind]);
    }
    if (profiler_trace_path()) record_sample(ts, kind, value);
}

profiler_region region_capture(bool blocking)
{
    auto & global           = state();
    thread_state & ts       = this_thread();
    const chunk * c         = ts.chunks.empty() ? nullptr : &ts.chunks.back();
    const bool own          = !ts.stack.empty() && (!c || ts.stack.size() > c->base);
    node * parent           = nullptr;
    std::uint64_t outer_run = 0;
    if (own)
    {
        parent    = ts.stack.back().task;
        outer_run = ts.stack.back().run;
    }
    else if (c)
    {
        parent    = c->parent;
        outer_run = c->run;
    }
    else
    {
        parent    = global.anchor.load(std::memory_order_acquire);
        outer_run = parent ? parent->current_run.load(std::memory_order_relaxed) : 0;
    }

    // Every region is a run of its own: the report takes the busiest thread of each, and counts it on
    // the line of the thread that started it within the enclosing run.
    const std::uint64_t run = global.started_runs.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        std::lock_guard<std::mutex> lock(global.regions_mutex);
        if (global.regions.size() < max_regions) global.regions.emplace(run, std::make_pair(outer_run, ts.index));
    }
    if (blocking && own) ts.started.push_back({ run, now_ns(), ts.stack.size() - 1, ts.stack.back().child_time });
    return { parent, run };
}

void region_finish(const profiler_region & region)
{
    thread_state & ts = this_thread();
    for (std::size_t i = ts.started.size(); i-- > 0;)
    {
        const auto started = ts.started[i];
        if (started.run != region.run) continue;
        ts.started.erase(ts.started.begin() + i);
        if (started.depth < ts.stack.size())
        {
            // The wait counts as child time, except for the parts already counted: the chunks
            // this thread ran itself as child tasks.
            frame & f                   = ts.stack[started.depth];
            const std::uint64_t elapsed = now_ns() - started.start;
            const std::uint64_t counted = f.child_time - started.child_time;
            const std::uint64_t wait    = elapsed > counted ? elapsed - counted : 0;
            f.child_time += wait;
            f.task->wait_total += wait;
            f.task->wait_count += 1;
            f.task->wait_max = std::max(f.task->wait_max, wait);
            f.task->wait_min = std::min(f.task->wait_min, wait);
        }
        return;
    }
}

void region_enter(const profiler_region & region)
{
    thread_state & ts = this_thread();
    chunk c;
    c.parent = static_cast<node *>(const_cast<void *>(region.parent));
    c.run    = region.run;
    c.base   = ts.stack.size();

    // A thread that waits inside other work (for example in a nested parallel call) may be given a
    // chunk of an unrelated region. Its time is not part of the tasks it interrupted: those are the
    // ones above the region's parent, or above the outermost chunk of the same region.
    const node * innermost = ts.stack.empty() ? nullptr : ts.stack.back().task;
    if (!ts.chunks.empty() && ts.stack.size() == ts.chunks.back().base) innermost = ts.chunks.back().parent;
    if (!ts.stack.empty() && innermost != c.parent)
    {
        c.interrupts = true;
        c.pause_from = 0;
        bool found   = false;
        for (std::size_t i = ts.stack.size(); i-- > 0;)
        {
            if (ts.stack[i].task == c.parent)
            {
                c.pause_from = i + 1;
                found        = true;
                break;
            }
        }
        for (std::size_t i = 0; !found && i < ts.chunks.size(); ++i)
        {
            if (ts.chunks[i].parent == c.parent)
            {
                c.pause_from = ts.chunks[i].base;
                break;
            }
        }
        c.start = now_ns();
    }
    ts.chunks.push_back(c);
}

void region_leave()
{
    thread_state & ts = this_thread();
    if (ts.chunks.empty()) return;
    const chunk c = ts.chunks.back();
    ts.chunks.pop_back();
    if (c.interrupts)
    {
        const std::uint64_t elapsed = now_ns() - c.start;
        for (std::size_t i = c.pause_from; i < c.base && i < ts.stack.size(); ++i) ts.stack[i].paused += elapsed;
    }
}

// Merged tree used by the report.
struct merged
{
    std::string name;
    merged * parent = nullptr;
    std::vector<std::unique_ptr<merged> > children;
    bool parallel                    = false;
    std::uint64_t count              = 0;
    std::uint64_t cpu                = 0;
    std::uint64_t min                = UINT64_MAX;
    std::uint64_t max                = 0;
    std::uint64_t peak[memory_kinds] = {};
    own_stats own;
    std::unordered_map<std::size_t, std::uint64_t> per_thread;

    struct run_time
    {
        std::uint64_t run;
        std::size_t thread;
        std::uint64_t time;
    };
    std::vector<run_time> runs;
    std::uint64_t unsplit_busiest = 0;
    std::uint64_t sequential_cpu  = 0; // runs outside any parallel region, which have no run entries
    std::uint64_t wait_total      = 0;
    std::uint64_t wait_count      = 0;
    std::uint64_t wait_max        = 0;
    std::uint64_t wait_min        = UINT64_MAX;
    std::uint64_t self_cpu        = 0;

    std::uint64_t wall        = 0;
    std::uint64_t busiest     = 0;
    std::uint64_t mean_thread = 0;
    std::uint64_t threads     = 0;
    std::uint64_t self        = 0;
    own_stats inclusive;
};

struct merged_tree
{
    merged root;
    own_stats outside;
    std::uint64_t task_runs = 0;
    std::size_t threads     = 0;
};

merged * merged_child(merged * parent, const std::string & name)
{
    for (auto & child : parent->children)
    {
        if (child->name == name) return child.get();
    }
    parent->children.push_back(std::make_unique<merged>());
    merged * created = parent->children.back().get();
    created->name    = name;
    created->parent  = parent;
    return created;
}

merged * resolve(merged_tree & tree, std::unordered_map<const node *, merged *> & map, const node * n)
{
    auto it = map.find(n);
    if (it != map.end()) return it->second;
    merged * parent = n->parent ? resolve(tree, map, n->parent) : (n->anchor ? resolve(tree, map, n->anchor) : &tree.root);
    merged * result = merged_child(parent, n->name);
    map.emplace(n, result);
    return result;
}

using region_map = std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::size_t> >;

const char * const waiting_row_name = "(waiting for parallel work)";

void finish(merged & m, const region_map & regions)
{
    for (auto & child : m.children) finish(*child, regions);

    m.wall        = m.cpu;
    m.busiest     = m.cpu;
    m.mean_thread = m.cpu;
    m.threads     = m.per_thread.size();
    if (m.parallel)
    {
        // Time per run and thread. Runs are numbered in start order, so a region comes after the run
        // it was started in and is folded into it first.
        std::map<std::uint64_t, std::unordered_map<std::size_t, std::uint64_t>, std::greater<std::uint64_t> > lines;
        for (const auto & entry : m.runs) lines[entry.run][entry.thread] += entry.time;

        m.busiest     = m.unsplit_busiest + m.sequential_cpu;
        m.mean_thread = m.unsplit_busiest + m.sequential_cpu;
        m.threads     = 0;
        for (const auto & run : lines)
        {
            std::uint64_t busiest = 0, sum = 0;
            for (const auto & line : run.second)
            {
                busiest = std::max(busiest, line.second);
                sum += line.second;
            }
            m.busiest += busiest;
            m.mean_thread += sum / run.second.size();
            m.threads = std::max<std::uint64_t>(m.threads, run.second.size());
        }

        // A run lasts as long as its busiest thread, and that time is spent on the line of the thread
        // that started the region, within the run it was started in.
        m.wall = m.unsplit_busiest + m.sequential_cpu;
        while (!lines.empty())
        {
            const auto it         = lines.begin();
            std::uint64_t busiest = 0;
            for (const auto & line : it->second) busiest = std::max(busiest, line.second);
            const auto region = regions.find(it->first);
            if (region != regions.end() && region->second.first != 0)
                lines[region->second.first][region->second.second] += busiest;
            else
                m.wall += busiest;
            lines.erase(it);
        }
    }

    // Without this row, time spent waiting for parallel regions whose chunks open no task would show
    // up nowhere.
    if (m.wait_total > 0)
    {
        merged * waiting = merged_child(&m, waiting_row_name);
        waiting->count   = m.wait_count;
        waiting->max     = m.wait_max;
        waiting->min     = m.wait_min;
        waiting->cpu     = m.wait_total;
        waiting->wall    = m.parallel && m.cpu > 0 ? static_cast<std::uint64_t>(double(m.wait_total) * double(m.wall) / double(m.cpu)) : m.wait_total;
        waiting->wall    = std::min(waiting->wall, m.wall);
        waiting->self    = waiting->wall;
        waiting->busiest = waiting->wall;
        waiting->mean_thread = waiting->wall;
        waiting->threads     = m.threads;
    }

    m.self      = m.parallel && m.cpu > 0 ? static_cast<std::uint64_t>(double(m.self_cpu) * double(m.wall) / double(m.cpu)) : m.self_cpu;
    m.self      = std::min(m.self, m.wall);
    m.inclusive = m.own;
    for (auto & child : m.children)
    {
        m.inclusive.add(child->inclusive);
        for (int k = 0; k < memory_kinds; ++k) m.peak[k] = std::max(m.peak[k], child->peak[k]);
    }
}

merged_tree build_tree()
{
    merged_tree tree;
    auto & global = state();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    std::unordered_map<const node *, merged *> map;
    for (const auto & ts : global.threads)
    {
        for (const node & n : ts->nodes)
        {
            merged * m = resolve(tree, map, &n);
            m->parallel |= n.in_parallel;
            m->count += n.count;
            m->cpu += n.total;
            if (n.count > 0)
            {
                m->min = std::min(m->min, n.min);
                m->max = std::max(m->max, n.max);
            }
            for (int k = 0; k < memory_kinds; ++k) m->peak[k] = std::max(m->peak[k], n.peak[k]);
            m->own.add(n.own);
            m->per_thread[ts->index] += n.total;
            for (const auto & run : n.runs) m->runs.push_back({ run.first, ts->index, run.second });
            m->self_cpu += n.self_total;
            m->sequential_cpu += n.sequential_total;
            m->wait_total += n.wait_total;
            m->wait_count += n.wait_count;
            m->wait_max        = std::max(m->wait_max, n.wait_max);
            m->wait_min        = std::min(m->wait_min, n.wait_min);
            m->unsplit_busiest = std::max(m->unsplit_busiest, n.unsplit_total);
            tree.task_runs += n.count;
        }
        if (!ts->nodes.empty()) tree.threads += 1;
        for (const auto & entry : ts->anchored) resolve(tree, map, entry.first)->own.add(entry.second);
        tree.outside.add(ts->outside);
    }
    region_map regions;
    {
        std::lock_guard<std::mutex> regions_lock(global.regions_mutex);
        regions = global.regions;
    }
    for (auto & child : tree.root.children) finish(*child, regions);
    return tree;
}

void flatten(const merged & m, std::int64_t parent, std::int64_t depth, std::vector<profiler_report_row> & rows)
{
    profiler_report_row row {};
    row.name           = m.name;
    row.parent         = parent;
    row.depth          = depth;
    row.parallel       = m.parallel;
    row.count          = m.count;
    row.time_ns        = m.wall;
    row.cpu_ns         = m.cpu;
    row.self_ns        = m.self;
    row.min_ns         = m.count > 0 ? m.min : 0;
    row.max_ns         = m.max;
    row.threads        = m.threads;
    row.busiest_ns     = m.busiest;
    row.mean_thread_ns = m.mean_thread;
    for (int k = 0; k < memory_kinds; ++k)
    {
        row.alloc_bytes[k] = m.inclusive.alloc_bytes[k];
        row.alloc_count[k] = m.inclusive.alloc_count[k];
        row.free_bytes[k]  = m.inclusive.free_bytes[k];
        row.free_count[k]  = m.inclusive.free_count[k];
        row.peak_bytes[k]  = m.peak[k];
    }
    for (int k = 0; k < transfer_kinds; ++k)
    {
        row.transfer_bytes[k] = m.inclusive.transfer_bytes[k];
        row.transfer_count[k] = m.inclusive.transfer_count[k];
    }
    const std::int64_t index = static_cast<std::int64_t>(rows.size());
    rows.push_back(std::move(row));
    for (const auto & child : m.children) flatten(*child, index, depth + 1, rows);
}

/// Cost of opening and closing one task, measured on a private state that publishes nothing.
double task_cost_ns()
{
    constexpr int runs = 20000;
    thread_state scratch;
    const char * name         = "calibration";
    const std::uint64_t start = now_ns();
    for (int i = 0; i < runs; ++i) end_task(scratch, begin_task(scratch, name, false, false), false);
    return double(now_ns() - start) / runs;
}

std::string format_bytes(std::uint64_t bytes)
{
    std::ostringstream out;
    const double value = static_cast<double>(bytes);
    if (bytes >= (1ull << 30))
        out << std::fixed << std::setprecision(2) << value / double(1ull << 30) << " GiB";
    else if (bytes >= (1ull << 20))
        out << std::fixed << std::setprecision(2) << value / double(1ull << 20) << " MiB";
    else if (bytes >= (1ull << 10))
        out << std::fixed << std::setprecision(2) << value / double(1ull << 10) << " KiB";
    else
        out << bytes << " B";
    return out.str();
}

std::string percent(std::uint64_t part, std::uint64_t whole)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << (whole > 0 ? 100.0 * double(part) / double(whole) : 0.0);
    return out.str();
}

std::string memory_line(const char * label, const std::uint64_t alloc_bytes, std::uint64_t alloc_count, std::uint64_t free_bytes,
                        std::uint64_t free_count)
{
    std::ostringstream out;
    out << label << "allocated " << format_bytes(alloc_bytes) << " in " << alloc_count << " blocks, freed " << format_bytes(free_bytes) << " in "
        << free_count << " blocks";
    return out.str();
}

void escape_json(std::ostream & out, const std::string & text)
{
    for (const char c : text)
    {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (static_cast<unsigned char>(c) < 0x20)
            out << ' ';
        else
            out << c;
    }
}

void write_trace(const char * path)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "ONEDAL PROFILER: cannot open " << path << " for writing the timeline" << std::endl;
        return;
    }
    auto & global = state();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    const auto micros                              = [&](std::uint64_t t) { return double(t > global.origin ? t - global.origin : 0) / 1e3; };
    static const char * memory_names[memory_kinds] = { "host memory", "device memory" };

    out << std::fixed << std::setprecision(3);
    out << "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n";
    out << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":0,\"tid\":0,\"args\":{\"name\":\"oneDAL\"}}";
    for (const auto & ts : global.threads)
    {
        out << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":" << ts->index << ",\"args\":{\"name\":\"thread " << ts->index << "\"}}";
        for (const auto & e : ts->events)
        {
            out << ",\n{\"name\":\"";
            escape_json(out, e.task->name);
            out << "\",\"cat\":\"" << (e.parallel ? "parallel" : "task") << "\",\"ph\":\"X\",\"pid\":0,\"tid\":" << ts->index
                << ",\"ts\":" << micros(e.start) << ",\"dur\":" << double(e.end - e.start) / 1e3 << "}";
        }
        for (const auto & s : ts->samples)
        {
            out << ",\n{\"name\":\"" << memory_names[s.kind] << "\",\"ph\":\"C\",\"pid\":0,\"ts\":" << micros(s.time)
                << ",\"args\":{\"MiB\":" << double(s.live) / double(1 << 20) << "}}";
        }
    }
    out << "\n]}\n";
    if (global.trace_dropped.load() > 0)
    {
        std::cerr << "ONEDAL PROFILER: the timeline is truncated, " << global.trace_dropped.load()
                  << " events were dropped; raise ONEDAL_PROFILER_TRACE_LIMIT to keep them" << std::endl;
    }
}

void report_at_exit()
{
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
    try
    {
#endif
        if (is_analyzer_enabled())
        {
            std::ostringstream report;
            profiler_print_report(report);
            profiler_log(nullptr, report.str());
        }
        if (const char * path = profiler_trace_path()) write_trace(path);
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
    }
    catch (std::exception & e)
    {
        std::cerr << "ONEDAL PROFILER: unexpected error while writing the report: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "ONEDAL PROFILER: unexpected error while writing the report" << std::endl;
    }
#endif
}

void print_memory_summary(std::ostream & out, const merged_tree & tree)
{
    auto & global = state();
    if (is_memory_tracking_enabled())
    {
        own_stats all = tree.outside;
        for (const auto & child : tree.root.children) all.add(child->inclusive);
        static const char * labels[memory_kinds] = { "  host:   ", "  device: " };
        out << "\nMemory\n";
        for (int k = 0; k < memory_kinds; ++k)
        {
            if (all.alloc_count[k] == 0 && all.free_count[k] == 0) continue;
            out << memory_line(labels[k], all.alloc_bytes[k], all.alloc_count[k], all.free_bytes[k], all.free_count[k]) << ", peak "
                << format_bytes(positive_part(global.process_peak[k].load())) << ", live now " << format_bytes(positive_part(global.live[k].load()))
                << '\n';
            if (tree.outside.alloc_count[k] > 0)
            {
                out << "          of which outside any task: " << format_bytes(tree.outside.alloc_bytes[k]) << " in " << tree.outside.alloc_count[k]
                    << " blocks\n";
            }
        }
        if (all.transfer_count[0] > 0 || all.transfer_count[1] > 0)
        {
            out << "  copies: " << format_bytes(all.transfer_bytes[0]) << " to the device in " << all.transfer_count[0] << " copies, "
                << format_bytes(all.transfer_bytes[1]) << " to the host in " << all.transfer_count[1] << " copies\n";
        }
        if (global.untracked_frees.load() > 0)
        {
            out << "  " << global.untracked_frees.load() << " frees of blocks allocated before tracking started were ignored\n";
        }
    }
}

} // namespace

DAAL_EXPORT void profiler_log(const char * once_key, const std::string & text)
{
    auto & s = sink();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (once_key && !s.every_call)
    {
        if (!s.seen.insert(once_key).second) return;
        if (!s.noted)
        {
            s.noted = true;
            s.out() << "ONEDAL PROFILER: each call site is logged once; set ONEDAL_PROFILER_OUTPUT=<file> to log every call\n";
        }
    }
    s.out() << text;
    s.out().flush();
}

DAAL_EXPORT int profiler_begin_task(const char * task_name, bool threading)
{
    if (threading && is_logger_enabled())
    {
        const std::string key = std::string("threading ") + task_name;
        profiler_log(key.c_str(), std::string("-----------------------------------------------------------------------------\n"
                                              "THREADING Profiler task started ")
                                      + task_name + "\n");
    }
    return begin_task(this_thread(), task_name, threading, true);
}

DAAL_EXPORT int profiler_begin_task_with_args(const char * task_name, const std::string & args)
{
    return begin_task(this_thread(), task_name, false, true, &args);
}

DAAL_EXPORT const profiler_region_hooks & profiler_threading_hooks()
{
    return region_hooks;
}

DAAL_EXPORT void profiler_end_task(int depth)
{
    end_task(this_thread(), depth, true);
}

DAAL_EXPORT void profiler_record_alloc(const void * ptr, std::size_t bytes, profiler_memory_kind kind)
{
    if (!ptr) return;
    const int k = static_cast<int>(kind);
    {
        auto & shard = shard_for(ptr);
        std::lock_guard<std::mutex> lock(shard.mutex);
        shard.blocks[ptr] = { bytes, k };
    }
    auto & ts       = this_thread();
    own_stats & own = charge_target(ts);
    own.alloc_bytes[k] += bytes;
    own.alloc_count[k] += 1;
    update_live(ts, k, static_cast<std::int64_t>(bytes));
}

DAAL_EXPORT void profiler_record_free(const void * ptr)
{
    if (!ptr) return;
    std::size_t bytes = 0;
    int k             = 0;
    {
        auto & shard = shard_for(ptr);
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.blocks.find(ptr);
        if (it == shard.blocks.end())
        {
            state().untracked_frees.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        bytes = it->second.first;
        k     = it->second.second;
        shard.blocks.erase(it);
    }
    auto & ts       = this_thread();
    own_stats & own = charge_target(ts);
    own.free_bytes[k] += bytes;
    own.free_count[k] += 1;
    update_live(ts, k, -static_cast<std::int64_t>(bytes));
}

DAAL_EXPORT void profiler_record_transfer(std::size_t bytes, profiler_transfer_kind kind)
{
    const int k     = static_cast<int>(kind);
    own_stats & own = charge_target(this_thread());
    own.transfer_bytes[k] += bytes;
    own.transfer_count[k] += 1;
}

DAAL_EXPORT std::vector<profiler_report_row> profiler_report()
{
    merged_tree tree = build_tree();
    std::vector<profiler_report_row> rows;
    for (const auto & child : tree.root.children) flatten(*child, -1, 0, rows);
    return rows;
}

DAAL_EXPORT void profiler_print_report(std::ostream & out)
{
    const merged_tree tree = build_tree();
    std::vector<profiler_report_row> rows;
    for (const auto & child : tree.root.children) flatten(*child, -1, 0, rows);
    auto & global = state();

    std::uint64_t total = 0;
    for (const auto & row : rows)
    {
        if (row.depth == 0) total += row.time_ns;
    }
    bool any_memory[memory_kinds] = {};
    bool any_transfer             = false;
    for (const auto & row : rows)
    {
        for (int k = 0; k < memory_kinds; ++k) any_memory[k] |= row.alloc_count[k] > 0 || row.free_count[k] > 0;
        any_transfer |= row.transfer_count[0] > 0 || row.transfer_count[1] > 0;
    }

    std::uint64_t total_cpu = 0;
    for (const auto & row : rows)
    {
        if (row.depth == 0) total_cpu += row.cpu_ns;
    }
    const double cost   = task_cost_ns();
    const auto overhead = static_cast<std::uint64_t>(cost * double(tree.task_runs));

    const bool tree_only = is_tree_only_report();
    out << "oneDAL profiler report\n";
    if (rows.empty())
    {
        out << "  no profiled task ran\n";
        if (!tree_only) print_memory_summary(out, tree);
        out << std::flush;
        return;
    }
    if (!tree_only)
    {
        out << "  profiled wall time " << format_time(total) << " on " << tree.threads << " threads, " << tree.task_runs << " task runs\n";
        out << "  profiler cost about " << format_time(static_cast<std::uint64_t>(cost)) << " per task run, " << format_time(overhead)
            << " of CPU time in total (" << percent(overhead, total_cpu) << "% of the profiled CPU time)\n";
        out << "  time: wall time, a parallel task taking its busiest thread in each run; self: time not in child tasks\n";
        if (any_memory[0] || any_memory[1])
            out << "  alloc: bytes allocated in the task and its children; peak: highest live memory above the task's start\n";
        out << '\n';
    }

    // Columns, then the tree.
    std::ostringstream header;
    header << std::setw(10) << "time" << std::setw(7) << "%" << std::setw(10) << "self" << std::setw(9) << "calls" << std::setw(10) << "avg"
           << std::setw(10) << "max" << std::setw(8) << "thr";
    if (any_memory[0]) header << std::setw(11) << "host alloc" << std::setw(11) << "host peak";
    if (any_memory[1]) header << std::setw(11) << "dev alloc" << std::setw(11) << "dev peak";
    if (any_transfer) header << std::setw(11) << "to dev" << std::setw(11) << "to host";
    header << "  task";
    out << header.str() << '\n';

    std::vector<bool> last_child(rows.size(), false);
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        last_child[i] = true;
        for (std::size_t j = i + 1; j < rows.size(); ++j)
        {
            if (rows[j].depth < rows[i].depth) break;
            if (rows[j].depth == rows[i].depth && rows[j].parent == rows[i].parent)
            {
                last_child[i] = false;
                break;
            }
        }
    }

    std::vector<std::string> prefixes(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const auto & row = rows[i];
        std::string prefix;
        if (row.parent >= 0)
        {
            const auto & parent_prefix = prefixes[row.parent];
            prefix                     = parent_prefix;
            if (!prefix.empty())
            {
                // Continue the parent's branch line only if the parent has siblings below it.
                prefix.replace(prefix.size() - 4, 4, last_child[row.parent] ? "    " : "|   ");
            }
            prefix += last_child[i] ? "`-- " : "|-- ";
        }
        prefixes[i] = prefix;

        out << std::setw(10) << format_time(row.time_ns) << std::setw(7) << percent(row.time_ns, total) << std::setw(10) << format_time(row.self_ns)
            << std::setw(9) << row.count << std::setw(10) << format_time(row.count > 0 ? row.cpu_ns / row.count : 0) << std::setw(10)
            << format_time(row.max_ns) << std::setw(8) << row.threads;
        if (any_memory[0]) out << std::setw(11) << format_bytes(row.alloc_bytes[0]) << std::setw(11) << format_bytes(row.peak_bytes[0]);
        if (any_memory[1]) out << std::setw(11) << format_bytes(row.alloc_bytes[1]) << std::setw(11) << format_bytes(row.peak_bytes[1]);
        if (any_transfer) out << std::setw(11) << format_bytes(row.transfer_bytes[0]) << std::setw(11) << format_bytes(row.transfer_bytes[1]);
        out << "  " << prefix << row.name;
        if (row.parallel && row.threads > 1 && row.mean_thread_ns > 0)
        {
            // Busiest thread over the average thread, per run of the region: 1.00 is perfectly balanced.
            const double imbalance = double(row.busiest_ns) / double(row.mean_thread_ns);
            out << "  [parallel, cpu " << format_time(row.cpu_ns) << ", imbalance " << std::fixed << std::setprecision(2) << imbalance << "]";
        }
        out << '\n';
    }

    if (tree_only)
    {
        out << std::flush;
        return;
    }

    // Hotspots: self time summed over every place a task name appears.
    struct hotspot
    {
        std::uint64_t self  = 0;
        std::uint64_t count = 0;
    };
    std::unordered_map<std::string, hotspot> by_name;
    for (const auto & row : rows)
    {
        by_name[row.name].self += row.self_ns;
        by_name[row.name].count += row.count;
    }
    std::vector<std::pair<std::string, hotspot> > hotspots(by_name.begin(), by_name.end());
    std::sort(hotspots.begin(), hotspots.end(), [](const auto & a, const auto & b) { return a.second.self > b.second.self; });
    out << "\nHotspots by self time\n";
    out << std::setw(10) << "self" << std::setw(7) << "%" << std::setw(9) << "calls" << "  task\n";
    for (std::size_t i = 0; i < hotspots.size() && i < 15; ++i)
    {
        out << std::setw(10) << format_time(hotspots[i].second.self) << std::setw(7) << percent(hotspots[i].second.self, total) << std::setw(9)
            << hotspots[i].second.count << "  " << hotspots[i].first << '\n';
    }

    print_memory_summary(out, tree);
    out << std::flush;
}

DAAL_EXPORT void profiler_reset()
{
    auto & global = state();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    for (auto & ts : global.threads)
    {
        ts->root.children.clear();
        ts->nodes.clear();
        ts->stack.clear();
        ts->chunks.clear();
        ts->started.clear();
        ts->anchored.clear();
        ts->outside = own_stats();
        ts->events.clear();
        ts->samples.clear();
    }
    global.anchor = nullptr;
    {
        std::lock_guard<std::mutex> regions_lock(global.regions_mutex);
        global.regions.clear();
    }
    for (int k = 0; k < memory_kinds; ++k)
    {
        global.region_peak[k]  = global.live[k].load();
        global.process_peak[k] = global.live[k].load();
    }
    global.trace_budget  = global_state::trace_limit();
    global.trace_dropped = 0;
}

} // namespace internal
} // namespace daal
