/* file: predict_types.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
#include "multinomial_naive_bayes/prediction/JPredictionMethod.h"
#include "classifier/prediction/JNumericTableInputId.h"
#include "classifier/prediction/JPredictionResultId.h"
#include "classifier/prediction/JModelInputId.h"

using namespace daal;
using namespace daal::algorithms;

#define jBatch   com_intel_daal_algorithms_ComputeMode_batchValue

#define DefaultDense com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionMethod_DefaultDense
#define FastCSR      com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionMethod_FastCSR

#define Data         com_intel_daal_algorithms_classifier_prediction_NumericTableInputId_Data

#define ModelInput   com_intel_daal_algorithms_classifier_prediction_ModelInputId_Model

#define Prediction   com_intel_daal_algorithms_classifier_prediction_PredictionResultId_Prediction

typedef multinomial_naive_bayes::prediction::Batch<float, multinomial_naive_bayes::prediction::defaultDense>     nb_pr_of_s_dd;
typedef multinomial_naive_bayes::prediction::Batch<double, multinomial_naive_bayes::prediction::defaultDense>    nb_pr_of_d_dd;
typedef multinomial_naive_bayes::prediction::Batch<float, multinomial_naive_bayes::prediction::fastCSR>          nb_pr_of_s_fc;
typedef multinomial_naive_bayes::prediction::Batch<double, multinomial_naive_bayes::prediction::fastCSR>         nb_pr_of_d_fc;
