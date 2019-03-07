/* file: parameter.cpp */
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

#include "svd/JParameter.h"

#include "common_helpers.h"
USING_COMMON_NAMESPACES()

/*
 * Class:     com_intel_daal_algorithms_svd_Parameter
 * Method:    cSetLeftSingularMatrixFormat
 * Signature:(JI)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_Parameter_cSetLeftSingularMatrixFormat
(JNIEnv *env, jobject thisObj, jlong addr, jint format)
{
    (*((svd::Parameter *)addr)).leftSingularMatrix = (svd::SVDResultFormat)format;
}

/*
 * Class:     com_intel_daal_algorithms_svd_Parameter
 * Method:    cSetRightSingularMatrixFormat
 * Signature:(JI)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_Parameter_cSetRightSingularMatrixFormat
(JNIEnv *env, jobject thisObj, jlong addr, jint format)
{
    (*((svd::Parameter *)addr)).rightSingularMatrix = (svd::SVDResultFormat)format;
}

/*
 * Class:     com_intel_daal_algorithms_svd_Parameter
 * Method:    cGetLeftSingularMatrixFormat
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_svd_Parameter_cGetLeftSingularMatrixFormat
(JNIEnv *env, jobject thisObj, jlong addr)
{
    return(jint)(*((svd::Parameter *)addr)).leftSingularMatrix;
}

/*
 * Class:     com_intel_daal_algorithms_svd_Parameter
 * Method:    cGetRightSingularMatrixFormat
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_svd_Parameter_cGetRightSingularMatrixFormat
(JNIEnv *env, jobject thisObj, jlong addr)
{
    return(jint)(*((svd::Parameter *)addr)).rightSingularMatrix;
}
