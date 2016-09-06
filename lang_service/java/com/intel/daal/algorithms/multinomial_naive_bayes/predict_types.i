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
