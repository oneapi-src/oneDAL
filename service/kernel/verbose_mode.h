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
/*
struct json
{
    template <typename... Args>
    static void print(Args... args)
    {
        write('{');
        int depth = json_obj(1, args...);
        for (int i = 0; i < depth; ++i) write('}');
    }

    template <typename... Args>
    static void println(Args... args)
    {
        print(args...);
        write('\n');
    }

private:
    static void write(const char * const str);
    static void write(const char c);
    static void write(const int i);
    static void write(const unsigned long u);
    static void write(const long long int i);
    static void write(const unsigned long long int u);
    static void write(const double d);

    // helper for predecessor of obj_end_t
    constexpr static bool is_next_end() { return false; }

    template <typename Value>
    constexpr static bool is_next_end(Value)
    {
        return false;
    }

    // helper for predecessor of obj_end_t
    template <typename Value, typename... Tail>
    constexpr static bool is_next_end(Value, Tail...)
    {
        return std::is_same<typename std::decay<Value>::type, obj_end_t>::value;
    }

    // helper to put ',' between fields but not between field and '}' and not between '}' and '}'
    template <typename... Tail>
    static void comma(Tail... tail)
    {
        write((sizeof...(Tail) > 0 && !is_next_end(tail...)) ? "," : "");
    }

    // json recursion end
    static int json_obj(int depth) { return depth; }

                // bool
    template <typename Value, typename... Tail>
    static auto json_obj(int depth, const char * key, const Value & val, Tail... tail) ->
        typename std::enable_if<std::is_same<bool, typename std::decay<Value>::type>::value, int>::type
    {
        write('"');
        write(key);
        write("\":");
        write(val ? "true" : "false");
        comma(tail...);
        return json_obj(depth, tail...);
    }

                // enum
    template <typename Value, typename... Tail>
    static auto json_obj(int depth, const char * key, const Value & val, Tail... tail) ->
        typename std::enable_if<std::is_enum<typename std::decay<Value>::type>::value, int>::type
    {
        write('"');
        write(key);
        write("\":\"enum ");
        write(static_cast<int>(val));
        write('"');
        comma(tail...);
        return json_obj(depth, tail...);
    }

                    // print struct by ptr common impl
    template <typename Tptr, typename... Tail>
    static int struct_by_ptr_impl(const Tptr val)
    {
        write('"');
        write(key);
        write("\":");
        if (val)
        {
            print("nClusters", val->nClusters, "maxIterations", val->maxIterations, "accuracyThreshold", val->accuracyThreshold, "gamma", val->gamma,
                  "distanceType", val->distanceType, "assignFlag", val->assignFlag);
        }
        else
        {
            write("\"nullptr\"");
        }
        comma(tail...);
        return json_obj(depth, tail...);
    }

    // algorithms::kmeans::Parameter * overload
    template <typename... Tail>
    static int json_obj(int depth, const char * key, const algorithms::kmeans::Parameter * const val, Tail... tail)
    {
        write('"');
        write(key);
        write("\":");
        if (val)
        {
            print("nClusters", val->nClusters, "maxIterations", val->maxIterations, "accuracyThreshold", val->accuracyThreshold, "gamma", val->gamma,
                  "distanceType", val->distanceType, "assignFlag", val->assignFlag);
        }
        else
        {
            write("\"nullptr\"");
        }
        comma(tail...);
        return json_obj(depth, tail...);
    }

    // NumericTable * overload
    template <typename... Tail>
    static int json_obj(int depth, const char * key, const data_management::NumericTable * const val, Tail... tail)
    {
        write('"');
        write(key);
        write("\":");
        if (val)
        {
            print("numberOfColumns", val->getNumberOfColumns(), "numberOfRows", val->getNumberOfRows(),
                  "dataLayout", val->getDataLayout(), "dataMemoryStatus", val->getDataMemoryStatus());
            // if verbose level 3 - show small part of array
        }
        else
        {
            write("\"nullptr\"");
        }
        comma(tail...);
        return json_obj(depth, tail...);
    }

    // NumericTable ** overload
    template <typename... Tail>
    static int json_obj(int depth, const char * key, const data_management::NumericTable * const * val, Tail... tail)
    {
        if (val)
        {
            return json_obj(depth, key, *val, tail...);
        }
        else
        {
            write('"');
            write(key);
            write("\":\"nullptr\"");
            comma(tail...);
            return json_obj(depth, tail...);
        }
    }

    // pointer
    template <typename Value, typename... Tail>
    static auto json_obj(int depth, const char * key, const Value & val, Tail... tail) ->
        typename std::enable_if<std::is_pointer<typename std::decay<Value>::type>::value, int>::type
    {
        write('"');
        write(key);
        write("\":\"");
        write(nullptr == val ? "nullptr" : "<address>");
        write('"');
        comma(tail...);
        return json_obj(depth, tail...);
    }

    // int, float...
    template <typename Value, typename... Tail>
    static auto json_obj(int depth, const char * key, const Value & val, Tail... tail) ->
        typename std::enable_if<!std::is_same<bool, typename std::decay<Value>::type>::value
                                    && (std::is_floating_point<typename std::decay<Value>::type>::value
                                        || std::is_integral<typename std::decay<Value>::type>::value),
                                int>::type
    {
        write('"');
        write(key);
        write("\":");
        write(val);
        comma(tail...);
        return json_obj(depth, tail...);
    }

                 // c-style string
    template <typename... Tail>
    static int json_obj(int depth, const char * key, const char * val, Tail... tail)
    {
        write('"');
        write(key);
        write("\":\"");
        write(val ? val : "nullptr");
        write("\"");
        comma(tail...);
        return json_obj(depth, tail...);
    }

    // general overload for objects
    template <typename Value, typename... Tail>
    static auto json_obj(int depth, const char * key, const Value & val, Tail... tail) ->
        typename std::enable_if<std::is_class<typename std::decay<Value>::type>::value, int>::type
    {
        write('"');
        write(key);
        write("\":\"<object>\"");
        comma(tail...);
        return json_obj(depth, tail...);
    }

    // add overloads for special objects: e.g. sizes for containers, shared_ptr is null and soo on

    // json object 'begin' a.k.a '{'
    template <typename... Tail>
    static int json_obj(int depth, const char * key, const obj_begin_t &, Tail... tail)
    {
        write('"');
        write(key);
        write("\":{");
        return json_obj(depth + 1, tail...);
    }

    // json object 'end' a.k.a '}'
    template <typename... Tail>
    static int json_obj(int depth, const obj_end_t &, Tail... tail) // what if next is end too?!
    {
        write('}');
        comma(tail...);
        return json_obj(depth - 1, tail...);
    }
};*/

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

    // NumericTable ** overload
    json & put(const char * const key, const data_management::NumericTable * const * const val)
    {
        if (val)
        {
            return put(key, *val);
        }
        else
        {
            write('"');
            write(key);
            write("\":\"nullptr\"");
        }
    }

    // NumericTable *
    // todo: mv2cpp
    json & put(const char * const key, const data_management::NumericTable * const val)
    {
        comma_if_needed();
        need_comma = true;

        write('"');
        write(key);
        write("\":");
        if (val)
        {
            need_comma = false;
            begin();
            put("numberOfColumns", val->getNumberOfColumns());
            put("numberOfRows", val->getNumberOfRows());
            put("dataLayout", val->getDataLayout());
            put("dataMemoryStatus", val->getDataMemoryStatus());
            // if verbose level 3 - show small part of array
            end();
        }
        else
        {
            write("\"nullptr\"");
        }

        return *this;
    }

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

    // all pointers
    json & put(const char * const key, const void * const val)
    {
        comma_if_needed();
        need_comma = true;
        write('"');
        write(key);
        write("\":\"");
        write(nullptr == val ? "nullptr\"" : "<address>\"");
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
        if (verbose_t::level) writer.put("time", begin).put("total", 1000.0 * double(std::clock() - start) / CLOCKS_PER_SEC).put(end);
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
    #define SHOW_STAT0(...)  ((void)0)
    #define SHOW_STAT1(...)  ((void)0)
    #define SHOW_STAT2(...)  ((void)0)
    #define SHOW_STAT3(...)  ((void)0)
    #define SHOW_STAT4(...)  ((void)0)
    #define SHOW_STAT5(...)  ((void)0)
    #define SHOW_STAT6(...)  ((void)0)
    #define SHOW_STAT7(...)  ((void)0)
    #define SHOW_STAT8(...)  ((void)0)
    #define SHOW_STAT9(...)  ((void)0)
    #define SHOW_STAT10(...) ((void)0)
#endif // buildwithverbose

} // namespace verbose_mode
} // namespace service
} // namespace daal

#endif
