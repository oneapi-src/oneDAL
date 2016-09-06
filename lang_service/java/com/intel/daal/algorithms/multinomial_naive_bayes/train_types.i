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

#include "JComputeMode.h"
#include "multinomial_naive_bayes/training/JTrainingMethod.h"
#include "classifier/training/JInputId.h"
#include "classifier/training/JTrainingResultId.h"

using namespace daal;
using namespace daal::algorithms;

#define jBatch           com_intel_daal_algorithms_ComputeMode_batchValue
#define offline_cm       com_intel_daal_algorithms_ComputeMode_offlineValue
#define online_cm        com_intel_daal_algorithms_ComputeMode_onlineValue
#define distributed_cm   com_intel_daal_algorithms_ComputeMode_distributedValue

#define DefaultDense com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingMethod_DefaultDense
#define FastCSR      com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingMethod_FastCSR

#define Data         com_intel_daal_algorithms_classifier_training_InputId_Data
#define Labels       com_intel_daal_algorithms_classifier_training_InputId_Labels

#define ModelResult  com_intel_daal_algorithms_classifier_training_TrainingResultId_Model

typedef multinomial_naive_bayes::training::Batch<float, multinomial_naive_bayes::training::defaultDense>     nb_tr_of_s_dd;
typedef multinomial_naive_bayes::training::Batch<double, multinomial_naive_bayes::training::defaultDense>    nb_tr_of_d_dd;
typedef multinomial_naive_bayes::training::Batch<float, multinomial_naive_bayes::training::fastCSR>          nb_tr_of_s_fc;
typedef multinomial_naive_bayes::training::Batch<double, multinomial_naive_bayes::training::fastCSR>         nb_tr_of_d_fc;
