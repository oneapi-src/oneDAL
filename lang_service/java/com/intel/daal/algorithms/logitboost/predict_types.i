/* file: predict_types.i */
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

#include "JComputeMode.h"
#include "logitboost/prediction/JPredictionMethod.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;

#define jBatch   com_intel_daal_algorithms_ComputeMode_batchValue

#define DefaultDense com_intel_daal_algorithms_logitboost_prediction_PredictionMethod_DefaultDense

typedef logitboost::prediction::Batch<float, logitboost::prediction::defaultDense>     lb_pr_of_s_dd;
typedef logitboost::prediction::Batch<double, logitboost::prediction::defaultDense>    lb_pr_of_d_dd;

typedef SharedPtr<logitboost::prediction::Batch<float, logitboost::prediction::defaultDense> >    sp_lb_pr_of_s_dd;
typedef SharedPtr<logitboost::prediction::Batch<double, logitboost::prediction::defaultDense> >   sp_lb_pr_of_d_dd;
