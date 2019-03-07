/* file: train_input.cpp */
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

#include "JComputeMode.h"
#include "JComputeStep.h"
#include "ridge_regression/training/JInput.h"
#include "ridge_regression/training/JDistributedStep2MasterInput.h"
#include "ridge_regression/training/JTrainingMethod.h"
#include "ridge_regression/training/JTrainingInputId.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::ridge_regression;

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_Input
 * Method:    cSetInput
 * Signature:(JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_Input_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<ridge_regression::training::Input>::set<ridge_regression::training::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_Input
 * Method:    cGetInput
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_Input_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<ridge_regression::training::Input>::get<ridge_regression::training::InputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_DistributedStep2MasterInput
 * Method:    cAddInput
 * Signature:(JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_DistributedStep2MasterInput_cAddInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong presAddr)
{
    jniInput<ridge_regression::training::DistributedInput<step2Master> >::
        add<ridge_regression::training::Step2MasterInputId, training::PartialResult>(inputAddr, id, presAddr);
}
