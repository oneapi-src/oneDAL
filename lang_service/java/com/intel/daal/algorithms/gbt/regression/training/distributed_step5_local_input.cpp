/* file: distributed_step5_local_input.cpp */
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
#include "com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::gbt::regression::training;

/*
* Class:     com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput
* Method:    cSetNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput_cSetNumericTable
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong dcAddr)
{
    jniInput<DistributedInput<step5Local> >::set<Step5LocalNumericTableInputId, NumericTable>(inputAddr, id, dcAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput
* Method:    cGetNumericTable
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput_cGetNumericTable
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step5Local> >::get<Step5LocalNumericTableInputId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput
* Method:    cSetDataCollection
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput_cSetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong dcAddr)
{
    jniInput<DistributedInput<step5Local> >::set<Step5LocalCollectionInputId, DataCollection>(inputAddr, id, dcAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput
* Method:    cAddNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput_cAddNumericTable
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<DistributedInput<step5Local> >::add<Step5LocalCollectionInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput
* Method:    cGetDataCollection
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_training_DistributedStep5LocalInput_cGetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step5Local> >::get<Step5LocalCollectionInputId, DataCollection>(inputAddr, id);
}
