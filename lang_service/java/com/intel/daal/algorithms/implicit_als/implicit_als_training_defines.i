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

#include "com_intel_daal_algorithms_implicit_als_training_NumericTableInputId.h"
#define dataId         com_intel_daal_algorithms_implicit_als_training_NumericTableInputId_dataId
#include "com_intel_daal_algorithms_implicit_als_training_ModelInputId.h"
#define inputModelId   com_intel_daal_algorithms_implicit_als_training_ModelInputId_inputModelId
#include "com_intel_daal_algorithms_implicit_als_training_TrainingResultId.h"
#define modelId        com_intel_daal_algorithms_implicit_als_training_TrainingResultId_modelId

#include "com_intel_daal_algorithms_implicit_als_training_TrainingMethod.h"
#define FastCSR             com_intel_daal_algorithms_implicit_als_training_TrainingMethod_fastCSRId
#define DefaultDense        com_intel_daal_algorithms_implicit_als_training_TrainingMethod_defaultDenseId

#include "com_intel_daal_algorithms_implicit_als_training_MasterInputId.h"
#define inputOfStep2FromStep1Id     com_intel_daal_algorithms_implicit_als_training_MasterInputId_inputOfStep2FromStep1Id

#include "com_intel_daal_algorithms_implicit_als_training_PartialModelInputId.h"
#define partialModelId              com_intel_daal_algorithms_implicit_als_training_PartialModelInputId_partialModelId

#include "com_intel_daal_algorithms_implicit_als_training_Step3LocalCollectionInputId.h"
#define partialModelBlocksToNodeId  com_intel_daal_algorithms_implicit_als_training_Step3LocalCollectionInputId_partialModelBlocksToNodeId
#define offsetId                    com_intel_daal_algorithms_implicit_als_training_Step3LocalNumericTableInputId_offsetId

#include "com_intel_daal_algorithms_implicit_als_training_Step4LocalNumericTableInputId.h"
#define partialDataId               com_intel_daal_algorithms_implicit_als_training_Step4LocalNumericTableInputId_partialDataId
#define inputOfStep4FromStep2Id     com_intel_daal_algorithms_implicit_als_training_Step4LocalNumericTableInputId_inputOfStep4FromStep2Id

#include "com_intel_daal_algorithms_implicit_als_training_Step4LocalPartialModelsInputId.h"
#define partialModelsId             com_intel_daal_algorithms_implicit_als_training_Step4LocalPartialModelsInputId_partialModelsId
