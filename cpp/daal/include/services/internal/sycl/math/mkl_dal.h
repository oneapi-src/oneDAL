/* file: mkl_dal.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_MATH_MKL_DAL_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_MATH_MKL_DAL_H__

#ifdef __clang__
    #define DISABLE_MKL_DAL_SYCL_WARNINGS_BEGIN() _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wreorder-ctor\"")
    #define DISABLE_MKL_DAL_SYCL_WARNINGS_END()   _Pragma("clang diagnostic pop")
#else
    #define DISABLE_MKL_DAL_SYCL_WARNINGS_BEGIN()
    #define DISABLE_MKL_DAL_SYCL_WARNINGS_END()
#endif

DISABLE_MKL_DAL_SYCL_WARNINGS_BEGIN()
#include "mkl_dal_sycl.hpp"
DISABLE_MKL_DAL_SYCL_WARNINGS_END()

#undef DISABLE_MKL_DAL_SYCL_WARNINGS_BEGIN
#undef DISABLE_MKL_DAL_SYCL_WARNINGS_END

#endif
