/* file: config_ref.h */
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

#ifndef __CONFIG_REF_H__
#define __CONFIG_REF_H__

#include "services/daal_defines.h"
#include "services/env_detect.h"

#include "src/externals/service_service_ref.h"
#include "src/externals/service_blas_ref.h"
#include "src/externals/service_lapack_ref.h"
#include "src/externals/service_math_ref.h"
#include "src/externals/service_rng_ref.h"
#include "src/externals/service_spblas_ref.h"
#include "src/externals/service_stat_ref.h"

namespace daal
{
namespace internal
{
template <typename fpType, CpuType cpu>
using BlasBackend = ref::OpenBlas<fpType, cpu>;

template <typename fpType, CpuType cpu>
using LapackBackend = ref::OpenBlasLapack<fpType, cpu>;

template <typename fpType, CpuType cpu>
using MathBackend = ref::RefMath<fpType, cpu>;

template <CpuType cpu>
using BaseRngBackend = ref::BaseRNG<cpu>;

template <typename fpType, CpuType cpu>
using RNGsBackend = ref::RNGs<fpType, cpu>;

using ServiceBackend = ref::RefService;

template <typename fpType, CpuType cpu>
using SpBlasBackend = ref::RefSpBlas<fpType, cpu>;

template <typename fpType, CpuType cpu>
using StatisticsBackend = ref::RefStatistics<fpType, cpu>;

} // namespace internal
} // namespace daal

#endif
