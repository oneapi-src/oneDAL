/* file: service_profiler.h */
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
//  Profiler for time and memory measurement of kernels
//--
*/
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <exception>
#include "services/daal_defines.h"
#include "services/library_version_info.h"

#ifdef _WIN32
    #define PRETTY_FUNCTION __FUNCSIG__
#else
    #define PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

#define __SERVICE_PROFILER_H__

// Some Windows builds compile DAAL without /EHsc, where a try block does not compile.
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    #define DAAL_PROFILER_EXCEPTIONS_ENABLED
#endif

#define DAAL_PROFILER_CONCAT2(x, y) x##y
#define DAAL_PROFILER_CONCAT(x, y)  DAAL_PROFILER_CONCAT2(x, y)
#define DAAL_PROFILER_UNIQUE_ID     __LINE__

#define DAAL_PROFILER_MACRO_1(name)                       daal::internal::profiler::start_task(#name)
#define DAAL_PROFILER_MACRO_2(name, queue)                daal::internal::profiler::start_task(#name, queue)
#define DAAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

// HEADER OUTPUT
#define DAAL_PROFILER_PRINT_HEADER()                                                                          \
    do                                                                                                        \
    {                                                                                                         \
        std::cerr << "-----------------------------------------------------------------------------" << '\n'; \
        std::cerr << "File: " << __FILE__ << ", Line: " << __LINE__ << '\n';                                  \
        if (daal::internal::is_service_debug_enabled())                                                       \
        {                                                                                                     \
            std::cerr << PRETTY_FUNCTION << '\n';                                                             \
        }                                                                                                     \
    } while (0)

// ARGS LOGGING
#define DAAL_PROFILER_LOG_ARGS(task_name, ...)                                  \
    do                                                                          \
    {                                                                           \
        DAAL_PROFILER_PRINT_HEADER();                                           \
        std::cerr << "Profiler task_name: " << #task_name << " Printed args: "; \
        daal::internal::profiler_log_named_args(#__VA_ARGS__, __VA_ARGS__);     \
        std::cerr << '\n';                                                      \
    } while (0)

#define DAAL_PROFILER_TASK_WITH_ARGS(task_name, ...)                                                                                          \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_profiler_enabled())                                                                                            \
        {                                                                                                                                     \
            if (daal::internal::is_logger_enabled())                                                                                          \
            {                                                                                                                                 \
                DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                                                               \
            }                                                                                                                                 \
            return daal::internal::profiler::start_task(#task_name);                                                                          \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

#define DAAL_PROFILER_TASK(...)                                                                                                               \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_profiler_enabled())                                                                                            \
        {                                                                                                                                     \
            if (daal::internal::is_logger_enabled())                                                                                          \
            {                                                                                                                                 \
                DAAL_PROFILER_PRINT_HEADER();                                                                                                 \
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl;                                                             \
            }                                                                                                                                 \
            return DAAL_PROFILER_GET_MACRO(__VA_ARGS__, DAAL_PROFILER_MACRO_2, DAAL_PROFILER_MACRO_1, FICTIVE)(__VA_ARGS__);                  \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

#define DAAL_PROFILER_THREADING_TASK(task_name)                                                                                               \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_profiler_enabled())                                                                                            \
        {                                                                                                                                     \
            return daal::internal::profiler::start_threading_task(#task_name);                                                                \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

#define DAAL_PROFILER_SERVICE_TASK(...)                                                                                                       \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_service_debug_enabled())                                                                                       \
        {                                                                                                                                     \
            if (daal::internal::is_logger_enabled())                                                                                          \
            {                                                                                                                                 \
                DAAL_PROFILER_PRINT_HEADER();                                                                                                 \
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl;                                                             \
            }                                                                                                                                 \
            return DAAL_PROFILER_GET_MACRO(__VA_ARGS__, DAAL_PROFILER_MACRO_2, DAAL_PROFILER_MACRO_1, FICTIVE)(__VA_ARGS__);                  \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

inline static constexpr int PROFILER_MODE_OFF       = 0;
inline static constexpr int PROFILER_MODE_LOGGER    = 1;
inline static constexpr int PROFILER_MODE_TRACER    = 2;
inline static constexpr int PROFILER_MODE_ANALYZER  = 3;
inline static constexpr int PROFILER_MODE_ALL_TOOLS = 4;
inline static constexpr int PROFILER_MODE_DEBUG     = 5;

namespace daal
{
struct profiler_region_hooks;

namespace internal
{

// Every shared object that includes this header reads the mode on its own, and they all get the
// same answer. The recorded data lives only in libonedal_core (service_profiler.cpp), so tasks from
// all shared objects go into one tree.
inline static int daal_verbose_mode()
{
    static const int cached = [] {
        const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
        int result               = PROFILER_MODE_OFF;
        if (verbose_str)
        {
            char * endptr = nullptr;
            errno         = 0;
            long parsed   = std::strtol(verbose_str, &endptr, 10);
            if (errno == 0 && endptr != verbose_str && *endptr == '\0' && parsed >= 0 && parsed <= 5)
            {
                result = static_cast<int>(parsed);
            }
        }
        return result;
    }();
    return cached;
}

/// Path of the timeline file requested through ONEDAL_PROFILER_TRACE, or nullptr.
inline static const char * profiler_trace_path()
{
    static const char * path = [] {
        const char * value = std::getenv("ONEDAL_PROFILER_TRACE");
        return (value && *value) ? value : nullptr;
    }();
    return path;
}

inline void profiler_log_named_args(const char * /*names*/) {}

template <typename T, typename... Rest>
inline void profiler_log_named_args(const char * names, const T & value, Rest &&... rest)
{
    const char * comma = strchr(names, ',');
    std::string name   = comma ? std::string(names, comma) : std::string(names);
    name.erase(name.begin(), std::find_if(name.begin(), name.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    std::cerr << name << ": " << value << "; ";
    if (comma) profiler_log_named_args(comma + 1, std::forward<Rest>(rest)...);
}

inline static bool is_service_debug_enabled()
{
    static const bool service_debug_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_DEBUG;
    }();
    return service_debug_value;
}

inline static bool is_logger_enabled()
{
    static const bool logger_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_LOGGER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return logger_value;
}

inline static bool is_tracer_enabled()
{
    static const bool verbose_value = [] {
        std::ios::sync_with_stdio(false);
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return verbose_value;
}

inline static bool is_analyzer_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return profiler_value;
}

inline static bool is_profiler_enabled()
{
    static const bool profiler_value = [] { return daal_verbose_mode() != PROFILER_MODE_OFF || profiler_trace_path() != nullptr; }();
    return profiler_value;
}

/// Whether allocations are recorded. Only the analyzer report and the timeline show them;
/// ONEDAL_PROFILER_MEMORY=0 turns them off.
inline static bool is_memory_tracking_enabled()
{
    static const bool value = [] {
        const char * memory = std::getenv("ONEDAL_PROFILER_MEMORY");
        const bool off      = memory && std::strcmp(memory, "0") == 0;
        return !off && (is_analyzer_enabled() || profiler_trace_path() != nullptr);
    }();
    return value;
}

inline void print_header()
{
    if (is_profiler_enabled())
    {
        daal::services::LibraryVersionInfo ver;
        std::cerr << "Major version:          " << ver.majorVersion << '\n';
        std::cerr << "Minor version:          " << ver.minorVersion << '\n';
        std::cerr << "Update version:         " << ver.updateVersion << '\n';
        std::cerr << "Product status:         " << ver.productStatus << '\n';
        std::cerr << "Build:                  " << ver.build << '\n';
        std::cerr << "Build revision:         " << ver.build_rev << '\n';
        std::cerr << "Name:                   " << ver.name << '\n';
        std::cerr << "Processor optimization: " << ver.processor << '\n';
        std::cerr << '\n';
    }
}

/// Memory an allocation lives in. USM shared memory counts as device memory.
enum class profiler_memory_kind : int
{
    host   = 0,
    device = 1
};

/// Direction of an explicit copy between host and device memory.
enum class profiler_transfer_kind : int
{
    host_to_device = 0,
    device_to_host = 1
};

/// Opens a task on the calling thread.
///
/// @param[in] task_name    Name of the task; copied the first time it is seen at a tree position
/// @param[in] threading    Whether the task is the body of a parallel region
///
/// @return The stack depth to pass to profiler_end_task()
DAAL_EXPORT int profiler_begin_task(const char * task_name, bool threading);

/// Closes the task opened at `depth` on the calling thread, and any task still open above it.
///
/// @param[in] depth    Value returned by profiler_begin_task()
DAAL_EXPORT void profiler_end_task(int depth);

/// Records an allocation, charged to the innermost task open on the calling thread.
///
/// @param[in] ptr      Address of the allocation; nullptr is ignored
/// @param[in] bytes    Size of the allocation
/// @param[in] kind     Memory the allocation lives in
DAAL_EXPORT void profiler_record_alloc(const void * ptr, std::size_t bytes, profiler_memory_kind kind);

/// Records a deallocation. Pointers whose allocation was not recorded are ignored.
///
/// @param[in] ptr      Address passed to the deallocation function
DAAL_EXPORT void profiler_record_free(const void * ptr);

/// Records an explicit copy between host and device memory.
///
/// @param[in] bytes    Size of the copy
/// @param[in] kind     Direction of the copy
DAAL_EXPORT void profiler_record_transfer(std::size_t bytes, profiler_transfer_kind kind);

/// One task of the merged task tree. Arrays are indexed by profiler_memory_kind or
/// profiler_transfer_kind; memory figures include the task's children.
struct profiler_report_row
{
    std::string name;
    std::int64_t parent; // index of the parent row, -1 for a top-level task
    std::int64_t depth;
    bool parallel;                // ran inside a parallel region
    std::uint64_t count;          // number of runs
    std::uint64_t time_ns;        // wall time; for a parallel task, along the busiest thread of each run
    std::uint64_t cpu_ns;         // time summed over all threads
    std::uint64_t self_ns;        // wall time not spent in child tasks
    std::uint64_t min_ns;         // shortest run
    std::uint64_t max_ns;         // longest run
    std::uint64_t threads;        // number of threads; for a parallel task, the most in one run of the region
    std::uint64_t busiest_ns;     // for a parallel task: the busiest thread's time, summed over runs
    std::uint64_t mean_thread_ns; // for a parallel task: the average thread's time, summed over runs
    std::uint64_t alloc_bytes[2];
    std::uint64_t alloc_count[2];
    std::uint64_t free_bytes[2];
    std::uint64_t free_count[2];
    std::uint64_t peak_bytes[2]; // highest live memory above the level at task start
    std::uint64_t transfer_bytes[2];
    std::uint64_t transfer_count[2];
};

/// Returns the merged task tree recorded so far, parents before children. Open tasks are skipped.
DAAL_EXPORT std::vector<profiler_report_row> profiler_report();

/// Writes the analyzer report for everything recorded so far.
///
/// @param[in] out  Stream to write to
DAAL_EXPORT void profiler_print_report(std::ostream & out);

/// Drops everything recorded so far. No task may be open on any thread.
DAAL_EXPORT void profiler_reset();

/// Callbacks the threading layer calls around every chunk of a parallel region.
DAAL_EXPORT const daal::profiler_region_hooks & profiler_threading_hooks();

class profiler_task
{
public:
    profiler_task() = default;
    explicit profiler_task(int depth) : depth_(depth) {}
    inline ~profiler_task();

    profiler_task(const profiler_task &)             = delete;
    profiler_task & operator=(const profiler_task &) = delete;

    profiler_task(profiler_task && other) noexcept : depth_(other.depth_) { other.depth_ = -1; }

    profiler_task & operator=(profiler_task && other) noexcept
    {
        if (this != &other)
        {
            depth_       = other.depth_;
            other.depth_ = -1;
        }
        return *this;
    }

private:
    int depth_ = -1;
};

class profiler
{
public:
    /// Starts a profiling task
    ///
    /// @param[in] task_name The name of the task, or nullptr for a task that records nothing
    ///
    /// @return An object that ends the task when destroyed. Invoked by the DAAL_PROFILER_TASK macro.
    inline static profiler_task start_task(const char * task_name)
    {
        if (!task_name) return profiler_task();
        return profiler_task(profiler_begin_task(task_name, false));
    }

    /// Starts a task that is the body of a parallel region. Invoked by DAAL_PROFILER_THREADING_TASK.
    ///
    /// @param[in] task_name The name of the task, or nullptr for a task that records nothing
    inline static profiler_task start_threading_task(const char * task_name)
    {
        if (!task_name) return profiler_task();
        return profiler_task(profiler_begin_task(task_name, true));
    }
};

inline profiler_task::~profiler_task()
{
    if (depth_ >= 0)
    {
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
        try
        {
#endif
            profiler_end_task(depth_);
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
        }
        catch (std::exception & e)
        {
            std::cerr << "DAAL KERNEL_PROFILER: unexpected error while ending a task: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "DAAL KERNEL_PROFILER: unexpected error while ending a task" << std::endl;
        }
#endif
    }
}

} // namespace internal
} // namespace daal
