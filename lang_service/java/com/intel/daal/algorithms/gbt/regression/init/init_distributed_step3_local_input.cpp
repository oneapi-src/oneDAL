/* file: init_distributed_step3_local_input.cpp */
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

#include <jni.h>
#include "daal.h"
#include "com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep3LocalInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::gbt::regression::init;

/*
* Class:     com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep3LocalInput
* Method:    cSetNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep3LocalInput_cSetNumericTable
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong dcAddr)
{
    jniInput<DistributedInput<step3Local> >::set<Step3LocalNumericTableInputId, NumericTable>(inputAddr, id, dcAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep3LocalInput
* Method:    cGetNumericTable
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep3LocalInput_cGetNumericTable
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step3Local> >::get<Step3LocalNumericTableInputId, NumericTable>(inputAddr, id);
}
