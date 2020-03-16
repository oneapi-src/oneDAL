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

#include <cstdio>    // std::printf
#include <cstdlib>   // std::getenv
#include <exception> // std::terminate

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
    case sse2: return "Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2)";
    case ssse3: return "Supplemental Streaming SIMD Extensions 3 (SSSE3)";
    case sse42: return "Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2)";
    case avx: return "Intel(R) Advanced Vector Extensions (Intel(R) AVX)";
    case avx2: return "Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)";
    case avx512_mic: return "Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512)";
    case avx512: return "Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512)";

    default: return "Unknown cpu type";
    };
}

void json::write(const char * const str)
{
    if (0 > std::printf("%s", str)) std::terminate();
}
void json::write(const char c)
{
    if (0 > std::printf("%c", c)) std::terminate();
}
void json::write(const int i)
{
    if (0 > std::printf("%d", i)) std::terminate();
}
void json::write(const unsigned long u)
{
    if (0 > std::printf("%lu", u)) std::terminate();
}
void json::write(const long long int i)
{
    if (0 > std::printf("%lld", i)) std::terminate();
}
void json::write(const unsigned long long int u)
{
    if (0 > std::printf("%llu", u)) std::terminate();
}
void json::write(const double d)
{
    if (0 > std::printf("%f", d)) std::terminate();
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
