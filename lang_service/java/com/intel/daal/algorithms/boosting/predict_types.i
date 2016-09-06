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
#include "boosting/prediction/JNumericTableInputId.h"
#include "boosting/prediction/JPredictionResultId.h"
#include "boosting/prediction/JModelInputId.h"

using namespace daal;
using namespace daal::algorithms;

#define jBatch   com_intel_daal_algorithms_ComputeMode_batchValue

#define Data com_intel_daal_algorithms_boosting_prediction_NumericTableInputId_Data

#define ModelInput com_intel_daal_algorithms_boosting_prediction_ModelInputId_Model

#define PredictionResults com_intel_daal_algorithms_boosting_prediction_PredictionResultId_PredictionResults

typedef boosting::prediction::Batch b_pr_of;
