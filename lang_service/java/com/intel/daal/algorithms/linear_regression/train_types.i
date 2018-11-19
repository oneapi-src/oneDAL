/* file: train_types.i */
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
using namespace daal::data_management;
using namespace daal::services;

typedef linear_regression::training::Batch<double, linear_regression::training::normEqDense> lr_t_of_d_dne;
typedef linear_regression::training::Batch<float,  linear_regression::training::normEqDense> lr_t_of_s_dne;
typedef linear_regression::training::Batch<double, linear_regression::training::qrDense>     lr_t_of_d_dqr;
typedef linear_regression::training::Batch<float,  linear_regression::training::qrDense>     lr_t_of_s_dqr;

typedef linear_regression::training::Online<double, linear_regression::training::normEqDense> lr_t_on_d_dne;
typedef linear_regression::training::Online<float,  linear_regression::training::normEqDense> lr_t_on_s_dne;
typedef linear_regression::training::Online<double, linear_regression::training::qrDense>     lr_t_on_d_dqr;
typedef linear_regression::training::Online<float,  linear_regression::training::qrDense>     lr_t_on_s_dqr;

typedef linear_regression::training::Distributed<step1Local, double, linear_regression::training::normEqDense> lr_t_dl_d_dne;
typedef linear_regression::training::Distributed<step1Local, float,  linear_regression::training::normEqDense> lr_t_dl_s_dne;
typedef linear_regression::training::Distributed<step1Local, double, linear_regression::training::qrDense>     lr_t_dl_d_dqr;
typedef linear_regression::training::Distributed<step1Local, float,  linear_regression::training::qrDense>     lr_t_dl_s_dqr;

typedef linear_regression::training::Distributed<step2Master, double, linear_regression::training::normEqDense> lr_t_dm_d_dne;
typedef linear_regression::training::Distributed<step2Master, float,  linear_regression::training::normEqDense> lr_t_dm_s_dne;
typedef linear_regression::training::Distributed<step2Master, double, linear_regression::training::qrDense>     lr_t_dm_d_dqr;
typedef linear_regression::training::Distributed<step2Master, float,  linear_regression::training::qrDense>     lr_t_dm_s_dqr;
