/* file: batch.cpp */
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
#include "math/relu/JBatch.h"
#include "math/relu/JMethod.h"
#include "daal.h"
#include "common_helpers.h"

#define FastCSRMethodValue com_intel_daal_algorithms_math_relu_Method_FastCSRMethodValue
USING_COMMON_NAMESPACES()
using namespace daal::algorithms::math;

/*
 * Class:     com_intel_daal_algorithms_math_relu_Batch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_relu_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<relu::Method, relu::Batch, relu::defaultDense, relu::fastCSR>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_math_relu_Batch
 * Method:    cSetResult
 * Signature:(JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_math_relu_Batch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<relu::Method, relu::Batch, relu::defaultDense, relu::fastCSR>::setResult<relu::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_math_relu_Batch
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_relu_Batch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<relu::Method, relu::Batch, relu::defaultDense, relu::fastCSR>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_math_relu_Batch
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_relu_Batch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<relu::Method, relu::Batch, relu::defaultDense, relu::fastCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_math_relu_Batch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_relu_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<relu::Method, relu::Batch, relu::defaultDense, relu::fastCSR>::getClone(prec, method, algAddr);
}
