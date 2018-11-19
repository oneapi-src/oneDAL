/* file: implicit_als_init_defines.i */
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
