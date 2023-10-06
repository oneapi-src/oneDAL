/* file: mkl_dal_utils.h */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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
//  Utility functions for DAL wrappers over Intel(R) MKL routines.
//--
*/

#ifndef __ONEAPI_INTERNAL_MKL_DAL_UTILS_H__
#define __ONEAPI_INTERNAL_MKL_DAL_UTILS_H__

#include "services/internal/sycl/math/types.h"
#include "services/internal/sycl/math/mkl_dal.h"

namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
namespace math
{
namespace interface1
{
inline ::oneapi::fpk::transpose to_fpk_transpose(const math::Transpose & trans)
{
    using fpk_transpose = ::oneapi::fpk::transpose;
    return trans == math::Transpose::Trans ? fpk_transpose::trans : fpk_transpose::nontrans;
}

inline ::oneapi::fpk::uplo to_fpk_uplo(const math::UpLo & uplo)
{
    using fpk_uplo = ::oneapi::fpk::uplo;
    return uplo == math::UpLo::Upper ? fpk_uplo::upper : fpk_uplo::lower;
}

} // namespace interface1

using interface1::to_fpk_transpose;
using interface1::to_fpk_uplo;

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
