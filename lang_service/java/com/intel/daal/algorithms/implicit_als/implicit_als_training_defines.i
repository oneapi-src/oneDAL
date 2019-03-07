/* file: implicit_als_training_defines.i */
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

#include "JComputeMode.h"
#include "JComputeStep.h"

#include "implicit_als/training/JModelInputId.h"
#include "implicit_als/training/JMasterInputId.h"
#include "implicit_als/training/JTrainingResultId.h"
#include "implicit_als/training/JNumericTableInputId.h"
#include "implicit_als/training/JDistributedPartialResultStep1Id.h"
#include "implicit_als/training/JDistributedPartialResultStep2Id.h"
#include "implicit_als/training/JDistributedPartialResultStep3Id.h"
#include "implicit_als/training/JDistributedPartialResultStep4Id.h"

#include "implicit_als/training/JPartialModelInputId.h"
#include "implicit_als/training/JStep3LocalCollectionInputId.h"
#include "implicit_als/training/JStep3LocalNumericTableInputId.h"
#include "implicit_als/training/JStep4LocalPartialModelsInputId.h"
#include "implicit_als/training/JStep4LocalNumericTableInputId.h"

#include "implicit_als/training/JTrainingMethod.h"

#include "common_defines.i"

#define dataId         com_intel_daal_algorithms_implicit_als_training_NumericTableInputId_dataId
#define inputModelId   com_intel_daal_algorithms_implicit_als_training_ModelInputId_inputModelId
#define modelId        com_intel_daal_algorithms_implicit_als_training_TrainingResultId_modelId

#define step1InputModelId        com_intel_daal_algorithms_implicit_als_training_Step1LocalInputId_step1InputModelId
#define inputOfStep3FromStep2Id  com_intel_daal_algorithms_implicit_als_training_Step3LocalInputId_inputOfStep3FromStep2Id

#define FastCSR             com_intel_daal_algorithms_implicit_als_training_TrainingMethod_fastCSRId
#define DefaultDense        com_intel_daal_algorithms_implicit_als_training_TrainingMethod_defaultDenseId

#define inputOfStep2FromStep1Id     com_intel_daal_algorithms_implicit_als_training_MasterInputId_inputOfStep2FromStep1Id

#define partialModelId              com_intel_daal_algorithms_implicit_als_training_PartialModelInputId_partialModelId

#define partialModelBlocksToNodeId  com_intel_daal_algorithms_implicit_als_training_Step3LocalCollectionInputId_partialModelBlocksToNodeId
#define offsetId                    com_intel_daal_algorithms_implicit_als_training_Step3LocalNumericTableInputId_offsetId

#define partialDataId               com_intel_daal_algorithms_implicit_als_training_Step4LocalNumericTableInputId_partialDataId
#define inputOfStep4FromStep2Id     com_intel_daal_algorithms_implicit_als_training_Step4LocalNumericTableInputId_inputOfStep4FromStep2Id

#define partialModelsId             com_intel_daal_algorithms_implicit_als_training_Step4LocalPartialModelsInputId_partialModelsId
