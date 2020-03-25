/* file: verbose_mode.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Verbose run-time library mode
//--
*/

#ifndef __VERBOSE_MODE_H__
#define __VERBOSE_MODE_H__

#include <type_traits> // compile-time info
#include <ctime>       // clock() function
#include "services/env_detect.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/kmeans/kmeans_types.h"

#include "service/kernel/daal_strings.h"

namespace daal
{
namespace service
{
namespace verbose_mode
{
struct obj_begin_t
{};
struct obj_end_t
{};

const char * cpuTypeToStr(const CpuType type);

static const obj_begin_t begin;
static const obj_end_t end;

// Static initialization for verbose level variable
struct verbose_t
{
    static verbose_t & getInstance()
    {
        static verbose_t instance;
        return instance;
    }

    int level;

private:
    verbose_t();
};

struct json
{
    json();

    // bool
    json & put(const char * const key, const bool val);

    // general enum
    template <typename Value>
    auto put(const char * const key, const Value & val) ->
        typename std::enable_if<std::is_enum<typename std::decay<Value>::type>::value, json &>::type
    {
        comma_if_needed();
        need_comma = true;
        write('"');
        write(key);
        write("\":\"enum ");
        write(static_cast<int>(val));
        write('"');
        return *this;
    }
    json & put(const char * const key, const data_management::NumericTableIface::StorageLayout);
    json & put(const char * const key, const data_management::NumericTableIface::MemoryStatus);
    json & put(const char * const key, const algorithms::kmeans::DistanceType);

    // char*
    json & put(const char * const key, const char * const str);

    // int, float
    template <typename Value>
    auto put(const char * const key, const Value & val) ->
        typename std::enable_if<!std::is_same<bool, typename std::decay<Value>::type>::value
                                    && (std::is_floating_point<typename std::decay<Value>::type>::value
                                        || std::is_integral<typename std::decay<Value>::type>::value),
                                json &>::type
    {
        comma_if_needed();
        need_comma = true;
        write_key(key);
        write(val);
        return *this;
    }

    // dispatcher for all pointers;
    // it will invoke concrete function overload if serealizer for such type is avaliable
    // function should be: void put(json &, const Type &)
    // we using SFINAE here to allow array of pointer to be unrolled by-compiler
    // todo: using Argument Dependend Lookup for possible keeping such functions together
    // (but we need writer class in being visible the same translation unit)
    template <typename ValPtr>
    auto put(const char * const key, const ValPtr p) ->
        typename std::enable_if<std::is_pointer<typename std::decay<ValPtr>::type>::value, json &>::type
    {
        comma_if_needed();
        need_comma = false;
        write_key(key);

        need_comma = false;
        begin();
        print_obj(p);
        end();

        need_comma = true;
        return *this;
    }

    // begin|end new nested object in json
    json & put(const char * const key, const obj_begin_t &);
    json & put(const obj_end_t &);

    void finalize();
    ~json();

private:
    // raw output
    static void write(const char * const str);
    static void write(const char c);
    static void write(const int i);
    static void write(const unsigned int u);
    static void write(const unsigned long u);
    static void write(const long long int i);
    static void write(const unsigned long long int u);
    static void write(const double d);

    void comma_if_needed();

    void begin();
    void end();

    void write_escape(const char * const str);
    void write_key(const char * const key);

    // pointer dispatcher
    template <typename ValPtr>
    auto print_obj(const ValPtr p) -> typename std::enable_if<std::is_pointer<typename std::decay<ValPtr>::type>::value>::type
    {
        if (p)
        {
            print_obj(*p);
        }
        else
        {
            write("\"ptr\":\"nullptr\"");
        }
    }

    // algorithms::kmeans::Parameter &
    void print_obj(const algorithms::kmeans::Parameter & val);
    // data_management::NumericTable &
    void print_obj(const data_management::NumericTable & val);
    // fallback to unknown object types
    void print_obj(...) { write("\"content\":\"unknown\""); }

    int depth;
    bool need_comma;
};

template <typename algorithmFPType>
constexpr const char * fpTypeToStr()
{
    return "unknown type";
}

template <>
constexpr const char * fpTypeToStr<float>()
{
    return "float";
}

template <>
constexpr const char * fpTypeToStr<double>()
{
    return "double";
}

template <typename algorithmFPType, CpuType cpu>
struct kernel_verbose_raii
{
    template <typename... Args>
    kernel_verbose_raii(const char * const file, Args... args) : file_name(file)
    {
        if (verbose_t::getInstance().level == 2)
        {
            writer.put("kernel file", file_name).put("algorithmFPType", fpTypeToStr<algorithmFPType>());
            writer.put("env", begin).put("cpu", cpuTypeToStr(cpu)).put(end);
            writer.put("args", begin);
            put(args...);
            writer.put(end);
        }
        if (verbose_t::getInstance().level) start = std::clock();
    }
    ~kernel_verbose_raii()
    {
        if (verbose_t::getInstance().level)
            writer.put("time", begin).put("total, msec", 1000.0 * double(std::clock() - start) / CLOCKS_PER_SEC).put(end);
    }

private:
    void put() {}

    template <typename Value, typename... Args>
    void put(const char * const key, const Value & value, Args... args)
    {
        writer.put(key, value);
        put(args...);
    }

    const char * const file_name;
    json writer;
    std::clock_t start = 0;
};

#define VERBOSE_BUILD_ENABLED 1

#if VERBOSE_BUILD_ENABLED

    // we can't use if(level) kernel_verbose_raii(...) because it will be scope
    #define SHOW_STAT0()     ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer(__FILE__);
    #define SHOW_STAT1(arg0) ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer(__FILE__, #arg0, arg0);
    #define SHOW_STAT2(arg0, arg1) \
        ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer(__FILE__, #arg0, arg0, #arg1, arg1);
    #define SHOW_STAT3(arg0, arg1, arg2) \
        ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer(__FILE__, #arg0, arg0, #arg1, arg1, #arg2, arg2);
    #define SHOW_STAT4(arg0, arg1, arg2, arg3)                                                                                                      \
        ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer(__FILE__, #arg0, arg0, #arg1, arg1, #arg2, arg2, #arg3, \
                                                                                            arg3);
    #define SHOW_STAT5(arg0, arg1, arg2, arg3, arg4)                                                                                                \
        ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer(__FILE__, #arg0, arg0, #arg1, arg1, #arg2, arg2, #arg3, \
                                                                                            arg3, #arg4, arg4);
    #define SHOW_STAT6(arg0, arg1, arg2, arg3, arg4, arg5)                                                                                          \
        ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer(__FILE__, #arg0, arg0, #arg1, arg1, #arg2, arg2, #arg3, \
                                                                                            arg3, #arg4, arg4, #arg5, arg5);
    #define SHOW_STAT7(arg0, arg1, arg2, arg3, arg4, arg5, arg6)                                                                                    \
        ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer(__FILE__, #arg0, arg0, #arg1, arg1, #arg2, arg2, #arg3, \
                                                                                            arg3, #arg4, arg4, #arg5, arg5, #arg6, arg6);
    #define SHOW_STAT8(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)                       \
        ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer( \
            __FILE__, #arg0, arg0, #arg1, arg1, #arg2, arg2, #arg3, arg3, #arg4, arg4, #arg5, arg5, #arg6, arg6, #arg7, arg7);
    #define SHOW_STAT9(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)                 \
        ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer( \
            __FILE__, #arg0, arg0, #arg1, arg1, #arg2, arg2, #arg3, arg3, #arg4, arg4, #arg5, arg5, #arg6, arg6, #arg7, arg7, arg8);
    #define SHOW_STAT10(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)          \
        ::daal::service::verbose_mode::kernel_verbose_raii<algorithmFPType, cpu> raii_timer( \
            __FILE__, #arg0, arg0, #arg1, arg1, #arg2, arg2, #arg3, arg3, #arg4, arg4, #arg5, arg5, #arg6, arg6, #arg7, arg7, arg8, arg9);

#else
    #define SHOW_STAT0(...)  ((void)0);
    #define SHOW_STAT1(...)  ((void)0);
    #define SHOW_STAT2(...)  ((void)0);
    #define SHOW_STAT3(...)  ((void)0);
    #define SHOW_STAT4(...)  ((void)0);
    #define SHOW_STAT5(...)  ((void)0);
    #define SHOW_STAT6(...)  ((void)0);
    #define SHOW_STAT7(...)  ((void)0);
    #define SHOW_STAT8(...)  ((void)0);
    #define SHOW_STAT9(...)  ((void)0);
    #define SHOW_STAT10(...) ((void)0);
#endif // VERBOSE_BUILD_ENABLED

} // namespace verbose_mode
} // namespace service
} // namespace daal

#endif
