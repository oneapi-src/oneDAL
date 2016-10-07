/* file: svd_types.i */
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

typedef svd::Batch<double, svd::defaultDense> svd_of_d_def;
typedef svd::Batch<float,  svd::defaultDense> svd_of_s_def;
typedef svd::Online <double, svd::defaultDense> svd_on_d_def;
typedef svd::Online <float,  svd::defaultDense> svd_on_s_def;

typedef svd::Distributed <step1Local, double, svd::defaultDense>  svd_dl1_d_def;
typedef svd::Distributed <step1Local, float, svd::defaultDense>   svd_dl1_s_def;
typedef svd::Distributed <step2Master, double, svd::defaultDense> svd_dm2_d_def;
typedef svd::Distributed <step2Master, float, svd::defaultDense>  svd_dm2_s_def;
typedef svd::Distributed <step3Local, double, svd::defaultDense>  svd_dl3_d_def;
typedef svd::Distributed <step3Local, float, svd::defaultDense>   svd_dl3_s_def;
