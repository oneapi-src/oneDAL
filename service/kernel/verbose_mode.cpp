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

    default: return "Unknown cpu type";
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
    write('"');
    write(key);
    write("\":");
    write(val ? "true" : "false");
    return *this;
}

json & json::put(const char * const key, const char * const str)
{
    comma_if_needed();
    need_comma = true;
    write('"');
    write(key);
    write("\":");
    write_escape(str);
    return *this;
}

json & json::put(const char * const key, const data_management::NumericTableIface::StorageLayout val)
{
    switch (val)
    {
    case data_management::NumericTableIface::StorageLayout::soa: put(key, "soa"); break;
    case data_management::NumericTableIface::StorageLayout::aos: put(key, "aos"); break;
    case data_management::NumericTableIface::StorageLayout::csrArray: put(key, "csrArray"); break;
    case data_management::NumericTableIface::StorageLayout::upperPackedSymmetricMatrix: put(key, "upperPackedSymmetricMatrix"); break;
    case data_management::NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix: put(key, "lowerPackedSymmetricMatrix"); break;
    case data_management::NumericTableIface::StorageLayout::upperPackedTriangularMatrix: put(key, "upperPackedTriangularMatrix"); break;
    case data_management::NumericTableIface::StorageLayout::lowerPackedTriangularMatrix: put(key, "lowerPackedTriangularMatrix"); break;
    case data_management::NumericTableIface::StorageLayout::arrow: put(key, "arrow"); break;
    case data_management::NumericTableIface::StorageLayout::layout_unknown: put(key, "layout_unknown"); break;

    default: put(key, "unknown"); break;
    }
    return *this;
}

json & json::put(const char * const key, const data_management::NumericTableIface::MemoryStatus val)
{
    switch (val)
    {
    case data_management::NumericTableIface::MemoryStatus::notAllocated: put(key, "notAllocated"); break;
    case data_management::NumericTableIface::MemoryStatus::userAllocated: put(key, "userAllocated"); break;
    case data_management::NumericTableIface::MemoryStatus::internallyAllocated: put(key, "internallyAllocated"); break;

    default: put(key, "unknown"); break;
    }
    return *this;
}

json & json::put(const char * const key, const algorithms::kmeans::DistanceType val)
{
    switch (val)
    {
    case algorithms::kmeans::euclidean: put(key, "euclidean"); break;

    default: put(key, "unknown"); break;
    }
    return *this;
}

json & json::put(const char * const key, const obj_begin_t &)
{
    comma_if_needed();
    need_comma = false;
    write('"');
    write(key);
    write("\":");
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

json::~json()
{
    finalize();
}

void json::write(const char * const str)
{
    if (0 > puts(str)) exit(1);
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

template <typename Parameter>
auto print_common_parameters_clusters(json & writer, const Parameter & val) -> decltype(val.nClusters, void())
{
    writer.put("nClusters", val.nClusters);
}
template <typename... Args>
void print_common_parameters_clusters(json &, Args...)
{}

template <typename Parameter>
auto print_common_parameters_iterations(json & writer, const Parameter & val) -> decltype(val.maxIterations, void())
{
    writer.put("maxIterations", val.maxIterations);
}
template <typename... Args>
void print_common_parameters_iterations(json &, Args...)
{}

template <typename Parameter>
auto print_common_parameters_accuracy(json & writer, const Parameter & val) -> decltype(val.accuracyThreshold, void())
{
    writer.put("accuracyThreshold", val.accuracyThreshold);
}
template <typename... Args>
void print_common_parameters_accuracy(json &, Args...)
{}

template <typename Parameter>
auto print_common_parameters_batchIndices(json & writer, const Parameter & val) -> decltype(val.batchIndices, void())
{
    writer.put("batchIndices", val.batchIndices);
}
template <typename... Args>
void print_common_parameters_batchIndices(json &, Args...)
{}

template <typename Parameter>
auto print_common_parameters_batchSize(json & writer, const Parameter & val) -> decltype(val.batchSize, void())
{
    writer.put("batchSize", val.batchSize);
}
template <typename... Args>
void print_common_parameters_batchSize(json &, Args...)
{}

template <typename Parameter>
void print_common_parameters(json & writer, const Parameter & val)
{
    print_common_parameters_clusters(writer, val);
    print_common_parameters_iterations(writer, val);
    print_common_parameters_accuracy(writer, val);
    print_common_parameters_batchIndices(writer, val);
    print_common_parameters_batchSize(writer, val);
}

void json::print_obj(const algorithms::kmeans::Parameter & val)
{
    print_common_parameters(*this, val);
    put("gamma", val.gamma).put("distanceType", val.distanceType).put("assignFlag", val.assignFlag);
}

// data_management::NumericTable &
void json::print_obj(const data_management::NumericTable & val)
{
    put("numberOfColumns", val.getNumberOfColumns()).put("numberOfRows", val.getNumberOfRows());
    put("dataLayout", val.getDataLayout()).put("dataMemoryStatus", val.getDataMemoryStatus());
    // TODO: if verbose level 3 - show small part of array
}

verbose_t::verbose_t()
{
    const char * const var = std::getenv("DAAL_VERBOSE");
    if (var)
    {
        if ('1' == var[0] && '\0' == var[1]) verbose_t::level = 1;
        if ('2' == var[0] && '\0' == var[1]) verbose_t::level = 2;
    }
    else
    {
        verbose_t::level = 0;
    }
}

int verbose_t::level;

static const ::daal::service::verbose_mode::verbose_t static_init_verbose_mode;

} // namespace verbose_mode
} // namespace service
} // namespace daal
