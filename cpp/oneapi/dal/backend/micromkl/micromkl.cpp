// /*******************************************************************************
// * Copyright 2021 Intel Corporation
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// *******************************************************************************/

// #include <daal/include/services/daal_defines.h>
// #include "oneapi/dal/backend/micromkl/micromkl.hpp"
// #include "oneapi/dal/backend/dispatcher.hpp"

// #define __MICROMKL_INCLUDE_GUARD__

// #include "oneapi/dal/backend/micromkl/macro.hpp"

// /* ================================== SYEVD ================================= */
// #define SYEVD_F_DECLARGS(Float) \
//     (const char* jobz,          \
//      const char* uplo,          \
//      const DAAL_INT* n,         \
//      Float* a,                  \
//      const DAAL_INT* lda,       \
//      Float* w,                  \
//      Float* work,               \
//      const DAAL_INT* lwork,     \
//      DAAL_INT* iwork,           \
//      const DAAL_INT* liwork,    \
//      DAAL_INT* info,            \
//      int ijobz,                 \
//      int iuplo)

// #define SYEVD_C_DECLARGS(Float) \
//     (char jobz,                 \
//      char uplo,                 \
//      std::int64_t n,            \
//      Float* a,                  \
//      std::int64_t lda,          \
//      Float* w,                  \
//      Float* work,               \
//      std::int64_t lwork,        \
//      std::int64_t* iwork,       \
//      std::int64_t liwork,       \
//      std::int64_t& info)

// #define SYEVD_F_CALLARGS (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, ijobz, iuplo)

// #define SYEVD_C_CALLARGS                   \
//     (&jobz,                                \
//      &uplo,                                \
//      reinterpret_cast<DAAL_INT*>(&n),      \
//      a,                                    \
//      reinterpret_cast<DAAL_INT*>(&lda),    \
//      w,                                    \
//      work,                                 \
//      reinterpret_cast<DAAL_INT*>(&lwork),  \
//      reinterpret_cast<DAAL_INT*>(iwork),   \
//      reinterpret_cast<DAAL_INT*>(&liwork), \
//      reinterpret_cast<DAAL_INT*>(&info),   \
//      1,                                    \
//      1)

// #ifdef ONEDAL_REF
// FUNC_TEMPLATE(unused, syevd, SYEVD_F_DECLARGS, SYEVD_C_DECLARGS, SYEVD_F_CALLARGS, SYEVD_C_CALLARGS)
// #else
// FUNC_TEMPLATE(fpk_lapack,
//               syevd,
//               SYEVD_F_DECLARGS,
//               SYEVD_C_DECLARGS,
//               SYEVD_F_CALLARGS,
//               SYEVD_C_CALLARGS)
// #endif
