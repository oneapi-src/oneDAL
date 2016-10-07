/* file: train_types.i */
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
