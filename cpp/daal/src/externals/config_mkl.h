/* file: config_mkl.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//  Template wrappers for common BLAS functions.
//--
*/

#ifndef __CONFIG_MKL_H__
#define __CONFIG_MKL_H__

#include "services/daal_defines.h"
#include "services/env_detect.h"

#include "src/externals/service_blas_mkl.h"
#include "src/externals/service_lapack_mkl.h"
#include "src/externals/service_math_mkl.h"
#include "src/externals/service_rng_mkl.h"
#include "src/externals/service_service_mkl.h"
#include "src/externals/service_spblas_mkl.h"
#include "src/externals/service_stat_mkl.h"

namespace daal
{
namespace internal
{
template <typename fpType, CpuType cpu>
using BlasBackend = mkl::MklBlas<fpType, cpu>;

template <typename fpType, CpuType cpu>
using LapackBackend = mkl::MklLapack<fpType, cpu>;

template <typename fpType, CpuType cpu>
using MathBackend = mkl::MklMath<fpType, cpu>;

template <CpuType cpu>
using BaseRngBackend = mkl::BaseRNG<cpu>;

template <typename fpType, CpuType cpu>
using RNGsBackend = mkl::RNGs<fpType, cpu>;

using ServiceBackend = mkl::MklService;

template <typename fpType, CpuType cpu>
using SpBlasBackend = mkl::MklSpBlas<fpType, cpu>;

template <typename fpType, CpuType cpu>
using StatisticsBackend = mkl::MklStatistics<fpType, cpu>;

} // namespace internal
} // namespace daal

#endif
