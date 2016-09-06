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
#include "stump/prediction/JPredictionMethod.h"

using namespace daal;
using namespace daal::algorithms;

#define jBatch   com_intel_daal_algorithms_ComputeMode_offlineValue
#define jOnline  com_intel_daal_algorithms_ComputeMode_onlineValue
#define jLocal   com_intel_daal_algorithms_ComputeMode_distributedLocalValue
#define jMaster  com_intel_daal_algorithms_ComputeMode_distributedMasterValue

#define DefaultDense com_intel_daal_algorithms_stump_prediction_PredictionMethod_DefaultDense

typedef stump::prediction::Batch<float, stump::prediction::defaultDense>     st_pr_of_s_dd;
typedef stump::prediction::Batch<double, stump::prediction::defaultDense>    st_pr_of_d_dd;
