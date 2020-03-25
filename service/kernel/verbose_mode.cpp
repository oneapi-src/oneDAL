/* file: verbose_mode.cpp */
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

#include <stdio.h>  // printf (C language)
#include <cstdlib>  // std::getenv
#include <stdlib.h> // exit (C language)

#include "verbose_mode.h"

namespace daal
{
namespace service
{
namespace verbose_mode
{
const char * cpuTypeToStr(const CpuType type)
{
    switch (type)
    {
    case sse2: return "Intel(R) SSE2";
    case ssse3: return "SSSE3";
    case sse42: return "Intel(R) SSE4.2";
    case avx: return "Intel(R) AVX";
    case avx2: return "Intel(R) AVX2";
    case avx512_mic: return "Intel(R) AVX-512";
    case avx512: return "Intel(R) AVX-512";

    default: return "Unknown";
    };
}

json::json() : depth(0), need_comma(false)
{
    begin();
}

json & json::put(const char * const key, const bool val)
{
    comma_if_needed();
    need_comma = true;
    write_key(key);
    write(val ? "true" : "false");
    return *this;
}

json & json::put(const char * const key, const char * const str)
{
    comma_if_needed();
    need_comma = true;
    write_key(key);
    write_escape(str);
    return *this;
}

#define VERBOSE_ENUM_FUNCTION1(TYPE, NAME0)        VERBOSE_ENUM_FUNC_PREFIX(TYPE) VERBOSE_ENUM_LIST1(TYPE, NAME0) VERBOSE_ENUM_FUNC_POSTFIX
#define VERBOSE_ENUM_FUNCTION2(TYPE, NAME0, NAME1) VERBOSE_ENUM_FUNC_PREFIX(TYPE) VERBOSE_ENUM_LIST2(TYPE, NAME0, NAME1) VERBOSE_ENUM_FUNC_POSTFIX
#define VERBOSE_ENUM_FUNCTION3(TYPE, NAME0, NAME1, NAME2) \
    VERBOSE_ENUM_FUNC_PREFIX(TYPE) VERBOSE_ENUM_LIST3(TYPE, NAME0, NAME1, NAME2) VERBOSE_ENUM_FUNC_POSTFIX
#define VERBOSE_ENUM_FUNCTION4(TYPE, NAME0, NAME1, NAME2, NAME3) \
    VERBOSE_ENUM_FUNC_PREFIX(TYPE) VERBOSE_ENUM_LIST4(TYPE, NAME0, NAME1, NAME2, NAME3) VERBOSE_ENUM_FUNC_POSTFIX
#define VERBOSE_ENUM_FUNCTION5(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4) \
    VERBOSE_ENUM_FUNC_PREFIX(TYPE) VERBOSE_ENUM_LIST5(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4) VERBOSE_ENUM_FUNC_POSTFIX
#define VERBOSE_ENUM_FUNCTION6(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5) \
    VERBOSE_ENUM_FUNC_PREFIX(TYPE) VERBOSE_ENUM_LIST6(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5) VERBOSE_ENUM_FUNC_POSTFIX
#define VERBOSE_ENUM_FUNCTION7(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6) \
    VERBOSE_ENUM_FUNC_PREFIX(TYPE) VERBOSE_ENUM_LIST7(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6) VERBOSE_ENUM_FUNC_POSTFIX
#define VERBOSE_ENUM_FUNCTION8(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7) \
    VERBOSE_ENUM_FUNC_PREFIX(TYPE) VERBOSE_ENUM_LIST8(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7) VERBOSE_ENUM_FUNC_POSTFIX
#define VERBOSE_ENUM_FUNCTION9(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7, NAME8) \
    VERBOSE_ENUM_FUNC_PREFIX(TYPE) VERBOSE_ENUM_LIST9(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7, NAME8) VERBOSE_ENUM_FUNC_POSTFIX
#define VERBOSE_ENUM_FUNCTION10(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7, NAME8, NAME9) \
    VERBOSE_ENUM_FUNC_PREFIX(TYPE)                                                                          \
    VERBOSE_ENUM_LIST10(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7, NAME8, NAME9) VERBOSE_ENUM_FUNC_POSTFIX

#define VERBOSE_ENUM_FUNC_PREFIX(TYPE)                       \
    json & json::put(const char * const key, const TYPE val) \
    {                                                        \
        switch (val)                                         \
        {
#define VERBOSE_ENUM_FUNC_POSTFIX        \
    default: put(key, "Unknown"); break; \
        }                                \
        return *this;                    \
        }

#define VERBOSE_ENUM_LIST1(TYPE, NAME0) \
    case TYPE::NAME0:                   \
        put(key, #NAME0);               \
        break;
#define VERBOSE_ENUM_LIST2(TYPE, NAME0, NAME1)               VERBOSE_ENUM_LIST1(TYPE, NAME0) VERBOSE_ENUM_LIST1(TYPE, NAME1)
#define VERBOSE_ENUM_LIST3(TYPE, NAME0, NAME1, NAME2)        VERBOSE_ENUM_LIST2(TYPE, NAME0, NAME1) VERBOSE_ENUM_LIST1(TYPE, NAME2)
#define VERBOSE_ENUM_LIST4(TYPE, NAME0, NAME1, NAME2, NAME3) VERBOSE_ENUM_LIST2(TYPE, NAME0, NAME1) VERBOSE_ENUM_LIST2(TYPE, NAME2, NAME3)
#define VERBOSE_ENUM_LIST5(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4) \
    VERBOSE_ENUM_LIST4(TYPE, NAME0, NAME1, NAME2, NAME3) VERBOSE_ENUM_LIST1(TYPE, NAME4)
#define VERBOSE_ENUM_LIST6(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5) \
    VERBOSE_ENUM_LIST5(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4) VERBOSE_ENUM_LIST1(TYPE, NAME5)
#define VERBOSE_ENUM_LIST7(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6) \
    VERBOSE_ENUM_LIST6(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5) VERBOSE_ENUM_LIST1(TYPE, NAME6)
#define VERBOSE_ENUM_LIST8(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7) \
    VERBOSE_ENUM_LIST7(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6) VERBOSE_ENUM_LIST1(TYPE, NAME7)
#define VERBOSE_ENUM_LIST9(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7, NAME8) \
    VERBOSE_ENUM_LIST8(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7) VERBOSE_ENUM_LIST1(TYPE, NAME8)
#define VERBOSE_ENUM_LIST10(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7, NAME8, NAME9) \
    VERBOSE_ENUM_LIST9(TYPE, NAME0, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7, NAME8) VERBOSE_ENUM_LIST1(TYPE, NAME9)

VERBOSE_ENUM_FUNCTION9(data_management::NumericTableIface::StorageLayout, soa, aos, csrArray, upperPackedSymmetricMatrix, lowerPackedSymmetricMatrix,
                       upperPackedTriangularMatrix, lowerPackedTriangularMatrix, arrow, layout_unknown)
VERBOSE_ENUM_FUNCTION3(data_management::NumericTableIface::MemoryStatus, notAllocated, userAllocated, internallyAllocated)
VERBOSE_ENUM_FUNCTION1(algorithms::kmeans::DistanceType, euclidean)

json & json::put(const char * const key, const obj_begin_t &)
{
    comma_if_needed();
    need_comma = false;
    write_key(key);
    begin();
    return *this;
}

json & json::put(const obj_end_t &)
{
    end();
    return *this;
}

void json::finalize()
{
    for (int i = 0; i < depth; ++i) write('}');
    depth = 0;
    write('\n');
}

void json::comma_if_needed()
{
    if (need_comma) write(',');
}

void json::begin()
{
    comma_if_needed();
    need_comma = false;
    write('{');
    ++depth;
}

void json::end()
{
    --depth;
    write('}');
}

void json::write_escape(const char * const str)
{
    write('"');
    for (const char * c = str; '\0' != *c; ++c)
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

void json::write_key(const char * const key)
{
    write('"');
    write(key);
    write("\":");
}

json::~json()
{
    finalize();
}

void json::write(const char * const str)
{
    if (0 > printf("%s", str)) exit(1);
}

void json::write(const char c)
{
    if (0 > printf("%c", c)) exit(1);
}

void json::write(const int i)
{
    if (0 > printf("%d", i)) exit(1);
}

void json::write(const unsigned int u)
{
    if (0 > printf("%u", u)) exit(1);
}

void json::write(const unsigned long u)
{
    if (0 > printf("%lu", u)) exit(1);
}

void json::write(const long long int i)
{
    if (0 > printf("%lld", i)) exit(1);
}

void json::write(const unsigned long long int u)
{
    if (0 > printf("%llu", u)) exit(1);
}

void json::write(const double d)
{
    if (0 > printf("%f", d)) exit(1);
}

// this macro also should support chainig like an original function and should looks like a real function
#define put_pair(NAME) put(#NAME, val.NAME)

#define print_common_fields_func(arg0) print_common_fields_##arg0

// Generate SFINAE functions for each possible field
#define DECLARE_DAAL_STRING_CONST(arg0)                                                                   \
    template <typename Parameter>                                                                         \
    auto print_common_fields_func(arg0)(json & writer, const Parameter & val)->decltype(val.arg0, void()) \
    {                                                                                                     \
        writer.put_pair(arg0);                                                                            \
    }                                                                                                     \
    template <typename... Args>                                                                           \
    void print_common_fields_func(arg0)(json &, Args...)                                                  \
    {}

DAAL_STRINGS_LIST()
#undef DECLARE_DAAL_STRING_CONST

template <typename Struct>
void print_common_fields(json & writer, const Struct & val)
{
#define DECLARE_DAAL_STRING_CONST(arg0) print_common_fields_func(arg0)(writer, val);
    DAAL_STRINGS_LIST()
#undef DECLARE_DAAL_STRING_CONST
}

void json::print_obj(const algorithms::kmeans::Parameter & val)
{
    print_common_fields(*this, val);
}

// data_management::NumericTable &
void json::print_obj(const data_management::NumericTable & val)
{
    put_pair(getNumberOfColumns()).put_pair(getNumberOfRows());
    put_pair(getDataLayout()).put_pair(getDataMemoryStatus());
    // TODO: if verbose level 3 - show small part of array
}

verbose_t::verbose_t()
{
    const char * const var = std::getenv("DAAL_VERBOSE");
    if (var)
    {
        if ('1' == var[0] && '\0' == var[1]) level = 1;
        if ('2' == var[0] && '\0' == var[1]) level = 2;
    }
    else
    {
        level = 0;
    }
}

} // namespace verbose_mode
} // namespace service
} // namespace daal
