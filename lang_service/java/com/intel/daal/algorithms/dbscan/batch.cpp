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
#include "daal.h"
#include "com_intel_daal_algorithms_dbscan_Batch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::dbscan;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cInit
(JNIEnv *, jobject, jint prec, jint method, jdouble epsilon, jlong minObservations)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::newObj(prec, method, epsilon, minObservations);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cSetResult
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<dbscan::Method, Batch, defaultDense>::setResult<dbscan::Result>(prec,method,algAddr,resultAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::getClone(prec,method,algAddr);
}
