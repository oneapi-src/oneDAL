/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#pragma once

#include "oneapi/dal/backend/dispatcher.hpp"

#define GEMM_PARAMETERS(Float)                                                                    \
    bool transa, bool transb, std::int64_t m, std::int64_t n, std::int64_t k, Float alpha,        \
        const Float *a, std::int64_t lda, const Float *b, std::int64_t ldb, Float beta, Float *c, \
        std::int64_t ldc

namespace oneapi::dal::backend::mkl {

template <typename Float, typename Cpu = void>
void gemm(GEMM_PARAMETERS(Float));

template <typename Float>
void gemm(const context_cpu &ctx, GEMM_PARAMETERS(Float));

} // namespace oneapi::dal::backend::mkl

#undef GEMM_PARAMETERS
