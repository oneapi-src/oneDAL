/* file: implicit_als_training_defines.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "com_intel_daal_algorithms_implicit_als_training_NumericTableInputId.h"
#define dataId com_intel_daal_algorithms_implicit_als_training_NumericTableInputId_dataId
#include "com_intel_daal_algorithms_implicit_als_training_ModelInputId.h"
#define inputModelId com_intel_daal_algorithms_implicit_als_training_ModelInputId_inputModelId
#include "com_intel_daal_algorithms_implicit_als_training_TrainingResultId.h"
#define modelId com_intel_daal_algorithms_implicit_als_training_TrainingResultId_modelId

#include "com_intel_daal_algorithms_implicit_als_training_TrainingMethod.h"
#define FastCSR      com_intel_daal_algorithms_implicit_als_training_TrainingMethod_fastCSRId
#define DefaultDense com_intel_daal_algorithms_implicit_als_training_TrainingMethod_defaultDenseId

#include "com_intel_daal_algorithms_implicit_als_training_MasterInputId.h"
#define inputOfStep2FromStep1Id com_intel_daal_algorithms_implicit_als_training_MasterInputId_inputOfStep2FromStep1Id

#include "com_intel_daal_algorithms_implicit_als_training_PartialModelInputId.h"
#define partialModelId com_intel_daal_algorithms_implicit_als_training_PartialModelInputId_partialModelId

#include "com_intel_daal_algorithms_implicit_als_training_Step3LocalCollectionInputId.h"
#define partialModelBlocksToNodeId com_intel_daal_algorithms_implicit_als_training_Step3LocalCollectionInputId_partialModelBlocksToNodeId
#define offsetId                   com_intel_daal_algorithms_implicit_als_training_Step3LocalNumericTableInputId_offsetId

#include "com_intel_daal_algorithms_implicit_als_training_Step4LocalNumericTableInputId.h"
#define partialDataId           com_intel_daal_algorithms_implicit_als_training_Step4LocalNumericTableInputId_partialDataId
#define inputOfStep4FromStep2Id com_intel_daal_algorithms_implicit_als_training_Step4LocalNumericTableInputId_inputOfStep4FromStep2Id

#include "com_intel_daal_algorithms_implicit_als_training_Step4LocalPartialModelsInputId.h"
#define partialModelsId com_intel_daal_algorithms_implicit_als_training_Step4LocalPartialModelsInputId_partialModelsId
