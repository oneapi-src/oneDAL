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
    verbose_t();
    static int level;
};

struct json
{
    // todo: mv2cpp
    json() : depth(1), need_comma(false) { write("{"); }

    // bool
    // todo: mv2cpp
    json & put(const char * const key, const bool & val)
    {
        comma_if_needed();
        need_comma = true;
        write('"');
        write(key);
        write("\":");
        write(val ? "true" : "false");
        return *this;
    }

    // enum
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

    // char*
    // todo: mv2cpp
    json & put(const char * const key, const char * const str)
    {
        comma_if_needed();
        need_comma = true;
        write('"');
        write(key);
        write("\":");
        write_escape(str);
        return *this;
    }

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
        write('"');
        write(key);
        write("\":");
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
        write('"');
        write(key);
        write("\":");

        need_comma = false;
        begin();
        json_print(*this, p);
        end();

        need_comma = true;
        return *this;
    }

    // todo: mv2cpp
    json & put(const char * const key, const obj_begin_t &)
    {
        comma_if_needed();
        need_comma = false;
        write('"');
        write(key);
        write("\":");
        begin();
        return *this;
    }

    // todo: mv2cpp
    json & put(const obj_end_t &)
    {
        end();
        return *this;
    }

    // todo: mv2cpp
    void finalize()
    {
        for (int i = 0; i < depth; ++i) write('}');
        depth = 0;
        write('\n');
    }

    // todo: mv2cpp
    ~json() { finalize(); }

private:
    // raw output
    static void write(const char * const str);
    static void write(const char c);
    static void write(const int i);
    static void write(const unsigned long u);
    static void write(const long long int i);
    static void write(const unsigned long long int u);
    static void write(const double d);

    // todo: mv2cpp
    void comma_if_needed()
    {
        if (need_comma) write(',');
    }

    void begin()
    {
        comma_if_needed();
        need_comma = false;
        write('{');
        ++depth;
    }

    void end()
    {
        --depth;
        write('}');
    }

    // todo: mv2cpp
    void write_escape(const char * const str)
    {
        write('"');
        for (const char * c = str; *c; ++c)
        {
            switch (*c)
            {
            case '"': write("\\\""); break;
            case '\\': write("\\\\"); break;
            case '/': write("\\/"); break;
            case '\b': write("\\b"); break;
            case '\f': write("\\f"); break;
            case '\n': write("\\n"); break;
            case '\r': write("\\r"); break;
            case '\t': write("\\t"); break;

            default: write(*c); break;
            }
        }
        write('"');
    }

    template <typename ValPtr>
    friend auto json_print(json & writer, const ValPtr p) -> typename std::enable_if<std::is_pointer<typename std::decay<ValPtr>::type>::value>::type;

    int depth;
    bool need_comma;
};

// pointer dispatcher
template <typename ValPtr>
auto json_print(json & writer, const ValPtr p) -> typename std::enable_if<std::is_pointer<typename std::decay<ValPtr>::type>::value>::type
{
    if (p)
    {
        json_print(writer, *p);
    }
    else
    {
        writer.write("\"ptr\":\"nullptr\"");
    }
}

// mv2cpp or use inline to break ODR
// algorithms::kmeans::Parameter &
inline void json_print(json & writer, const algorithms::kmeans::Parameter & val)
{
    writer.put("nClusters", val.nClusters).put("maxIterations", val.maxIterations).put("accuracyThreshold", val.accuracyThreshold);
    writer.put("gamma", val.gamma).put("distanceType", val.distanceType).put("assignFlag", val.assignFlag);
}

// data_management::NumericTable &
// todo: mv2cpp
inline void json_print(json & writer, const data_management::NumericTable & val)
{
    writer.put("numberOfColumns", val.getNumberOfColumns()).put("numberOfRows", val.getNumberOfRows());
    writer.put("dataLayout", val.getDataLayout()).put("dataMemoryStatus", val.getDataMemoryStatus());
    // if verbose level 3 - show small part of array
}

inline void json_print(json & writer, ...) {}

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

template <>
constexpr const char * fpTypeToStr<int>()
{
    return "int";
}

template <typename algorithmFPType, CpuType cpu>
struct kernel_verbose_raii
{
    template <typename... Args>
    kernel_verbose_raii(const char * const file, Args... args) : file_name(file)
    {
        if (verbose_t::level == 2)
        {
            writer.put("kernel file", file_name).put("algorithmFPType", fpTypeToStr<algorithmFPType>());
            writer.put("env", begin).put("cpu", cpuTypeToStr(cpu)).put(end);
            writer.put("args", begin);
            put(args...);
            writer.put(end);
        }
        if (verbose_t::level) start = std::clock();
    }
    ~kernel_verbose_raii()
    {
        if (verbose_t::level) writer.put("time", begin).put("total, msec", 1000.0 * double(std::clock() - start) / CLOCKS_PER_SEC).put(end);
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

#define buildwithverbose 1

#if buildwithverbose

    // we can't use if(verbose::level) kernel_verbose_raii(...) because it will be scope
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

    // todo: #define SHOW_STAT(...) if(verbose::level) verbose_unroll_args(__VA_ARGS__);
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
#endif // buildwithverbose

} // namespace verbose_mode
} // namespace service
} // namespace daal

#endif
