/* file: lowordermoments_types.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
