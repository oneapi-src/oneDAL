/* file: input.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_bacon_outlier_detection_Input */

#include "daal.h"
#include "bacon_outlier_detection/JInput.h"
#include "bacon_outlier_detection/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::bacon_outlier_detection;

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Input_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<bacon_outlier_detection::Input>::set<bacon_outlier_detection::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Input
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Input_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<bacon_outlier_detection::Input>::get<bacon_outlier_detection::InputId, NumericTable>(inputAddr, id);
}
