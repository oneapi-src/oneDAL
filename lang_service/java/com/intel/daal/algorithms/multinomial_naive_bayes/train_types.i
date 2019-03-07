/* file: train_types.i */
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
