/* file: batch.cpp */
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
#include "pivoted_qr_types.i"

#include "pivoted_qr/JBatch.h"
#include "pivoted_qr/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::pivoted_qr;

/*
 * Class:     com_intel_daal_algorithms_cholesky_Batch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<pivoted_qr::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
 * Class:     Java_com_intel_daal_algorithms_pivoted_1qr_Batch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Batch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<pivoted_qr::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     Java_com_intel_daal_algorithms_pivoted_1qr_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Batch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<pivoted_qr::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     Java_com_intel_daal_algorithms_pivoted_1qr_Batch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Batch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<pivoted_qr::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     Java_com_intel_daal_algorithms_pivoted_1qr_Batch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Batch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<pivoted_qr::Method, Batch, defaultDense>::setResult<pivoted_qr::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     Java_com_intel_daal_algorithms_pivoted_1qr_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<pivoted_qr::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
