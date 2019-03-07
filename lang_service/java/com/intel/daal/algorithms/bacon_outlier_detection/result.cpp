/* file: result.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_bacon_outlier_detection_Result */

#include "daal.h"
#include "bacon_outlier_detection/JResult.h"
#include "bacon_outlier_detection/JResultId.h"
#include "bacon_outlier_detection/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::bacon_outlier_detection;

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<bacon_outlier_detection::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Result
 * Method:    cGetResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<bacon_outlier_detection::Result>::
        get<bacon_outlier_detection::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Result
 * Method:    cSetResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<bacon_outlier_detection::Result>::
        set<bacon_outlier_detection::ResultId, NumericTable>(resAddr, id, ntAddr);
}
