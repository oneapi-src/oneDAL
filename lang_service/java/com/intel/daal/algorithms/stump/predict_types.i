/* file: predict_types.i */
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
