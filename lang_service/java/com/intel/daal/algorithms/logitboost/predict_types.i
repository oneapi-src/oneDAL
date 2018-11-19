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
