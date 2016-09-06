/* file: lowordermoments_types.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "daal.h"

using namespace daal;
using namespace daal::algorithms;

#include "low_order_moments/JMethod.h"

#define DefaultDense    com_intel_daal_algorithms_low_order_moments_Method_DefaultDense
#define SinglePassDense com_intel_daal_algorithms_low_order_moments_Method_SinglePassDense
#define SumDense        com_intel_daal_algorithms_low_order_moments_Method_SumDense
#define FastCSR         com_intel_daal_algorithms_low_order_moments_Method_FastCSR
#define SinglePassCSR   com_intel_daal_algorithms_low_order_moments_Method_SinglePassCSR
#define SumCSR          com_intel_daal_algorithms_low_order_moments_Method_SumCSR

typedef low_order_moments::Batch<float, low_order_moments::defaultDense>     lom_of_s_dd;
typedef low_order_moments::Batch<float, low_order_moments::singlePassDense>  lom_of_s_pd;
typedef low_order_moments::Batch<float, low_order_moments::sumDense>         lom_of_s_sd;
typedef low_order_moments::Batch<double, low_order_moments::defaultDense>    lom_of_d_dd;
typedef low_order_moments::Batch<double, low_order_moments::singlePassDense> lom_of_d_pd;
typedef low_order_moments::Batch<double, low_order_moments::sumDense>        lom_of_d_sd;
typedef low_order_moments::Batch<float, low_order_moments::fastCSR>          lom_of_s_dc;
typedef low_order_moments::Batch<float, low_order_moments::singlePassCSR>    lom_of_s_pc;
typedef low_order_moments::Batch<float, low_order_moments::sumCSR>           lom_of_s_sc;
typedef low_order_moments::Batch<double, low_order_moments::fastCSR>         lom_of_d_dc;
typedef low_order_moments::Batch<double, low_order_moments::singlePassCSR>   lom_of_d_pc;
typedef low_order_moments::Batch<double, low_order_moments::sumCSR>          lom_of_d_sc;

typedef low_order_moments::Online<float, low_order_moments::defaultDense>     lom_on_s_dd;
typedef low_order_moments::Online<float, low_order_moments::singlePassDense>  lom_on_s_pd;
typedef low_order_moments::Online<float, low_order_moments::sumDense>         lom_on_s_sd;
typedef low_order_moments::Online<double, low_order_moments::defaultDense>    lom_on_d_dd;
typedef low_order_moments::Online<double, low_order_moments::singlePassDense> lom_on_d_pd;
typedef low_order_moments::Online<double, low_order_moments::sumDense>        lom_on_d_sd;
typedef low_order_moments::Online<float, low_order_moments::fastCSR>          lom_on_s_dc;
typedef low_order_moments::Online<float, low_order_moments::singlePassCSR>    lom_on_s_pc;
typedef low_order_moments::Online<float, low_order_moments::sumCSR>           lom_on_s_sc;
typedef low_order_moments::Online<double, low_order_moments::fastCSR>         lom_on_d_dc;
typedef low_order_moments::Online<double, low_order_moments::singlePassCSR>   lom_on_d_pc;
typedef low_order_moments::Online<double, low_order_moments::sumCSR>          lom_on_d_sc;

typedef low_order_moments::Distributed<step1Local, float, low_order_moments::defaultDense>     lom_dl_s_dd;
typedef low_order_moments::Distributed<step1Local, float, low_order_moments::singlePassDense>  lom_dl_s_pd;
typedef low_order_moments::Distributed<step1Local, float, low_order_moments::sumDense>         lom_dl_s_sd;
typedef low_order_moments::Distributed<step1Local, double, low_order_moments::defaultDense>    lom_dl_d_dd;
typedef low_order_moments::Distributed<step1Local, double, low_order_moments::singlePassDense> lom_dl_d_pd;
typedef low_order_moments::Distributed<step1Local, double, low_order_moments::sumDense>        lom_dl_d_sd;
typedef low_order_moments::Distributed<step1Local, float, low_order_moments::fastCSR>          lom_dl_s_dc;
typedef low_order_moments::Distributed<step1Local, float, low_order_moments::singlePassCSR>    lom_dl_s_pc;
typedef low_order_moments::Distributed<step1Local, float, low_order_moments::sumCSR>           lom_dl_s_sc;
typedef low_order_moments::Distributed<step1Local, double, low_order_moments::fastCSR>         lom_dl_d_dc;
typedef low_order_moments::Distributed<step1Local, double, low_order_moments::singlePassCSR>   lom_dl_d_pc;
typedef low_order_moments::Distributed<step1Local, double, low_order_moments::sumCSR>          lom_dl_d_sc;

typedef low_order_moments::Distributed<step2Master, float, low_order_moments::defaultDense>     lom_dm_s_dd;
typedef low_order_moments::Distributed<step2Master, float, low_order_moments::singlePassDense>  lom_dm_s_pd;
typedef low_order_moments::Distributed<step2Master, float, low_order_moments::sumDense>         lom_dm_s_sd;
typedef low_order_moments::Distributed<step2Master, double, low_order_moments::defaultDense>    lom_dm_d_dd;
typedef low_order_moments::Distributed<step2Master, double, low_order_moments::singlePassDense> lom_dm_d_pd;
typedef low_order_moments::Distributed<step2Master, double, low_order_moments::sumDense>        lom_dm_d_sd;
typedef low_order_moments::Distributed<step2Master, float, low_order_moments::fastCSR>          lom_dm_s_dc;
typedef low_order_moments::Distributed<step2Master, float, low_order_moments::singlePassCSR>    lom_dm_s_pc;
typedef low_order_moments::Distributed<step2Master, float, low_order_moments::sumCSR>           lom_dm_s_sc;
typedef low_order_moments::Distributed<step2Master, double, low_order_moments::fastCSR>         lom_dm_d_dc;
typedef low_order_moments::Distributed<step2Master, double, low_order_moments::singlePassCSR>   lom_dm_d_pc;
typedef low_order_moments::Distributed<step2Master, double, low_order_moments::sumCSR>          lom_dm_d_sc;
