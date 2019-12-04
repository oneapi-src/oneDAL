/* file: batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "daal.h"
#include "com_intel_daal_algorithms_dbscan_Batch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::dbscan;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cInit(JNIEnv *, jobject, jint prec, jint method, jdouble epsilon,
                                                                          jlong minObservations)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::newObj(prec, method, epsilon, minObservations);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cInitParameter(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                   jint method)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cGetInput(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cGetResult(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cSetResult(JNIEnv *, jobject, jlong algAddr, jint prec, jint method,
                                                                              jlong resultAddr)
{
    jniBatch<dbscan::Method, Batch, defaultDense>::setResult<dbscan::Result>(prec, method, algAddr, resultAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Batch_cClone(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dbscan::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
