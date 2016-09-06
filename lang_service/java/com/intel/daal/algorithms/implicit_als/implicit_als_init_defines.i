/* file: implicit_als_init_defines.i */
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

#include "JComputeMode.h"
#include "JComputeStep.h"

#include "implicit_als/training/init/JInitInputId.h"
#include "implicit_als/training/init/JInitResultId.h"
#include "implicit_als/training/init/JInitPartialResultId.h"
#include "implicit_als/training/init/JInitMethod.h"

#include "common_defines.i"


#define initDataId                com_intel_daal_algorithms_implicit_als_training_init_InitInputId_Data
#define initModelId               com_intel_daal_algorithms_implicit_als_training_init_InitResultId_modelId
#define initPartialModelId        com_intel_daal_algorithms_implicit_als_training_init_InitPartialResultId_partialModelId

#define InitFastCSR       com_intel_daal_algorithms_implicit_als_training_init_InitMethod_fastCSRId
#define InitDefaultDense  com_intel_daal_algorithms_implicit_als_training_init_InitMethod_defaultDenseId
