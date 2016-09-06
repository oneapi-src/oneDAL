/* file: parameter.cpp */
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

#include <jni.h>
#include "svd_types.i"

#include "svd/JParameter.h"


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
