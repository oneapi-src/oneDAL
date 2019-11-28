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
#include "com_intel_daal_algorithms_pca_transform_TransformBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::pca::transform;

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformBatch_cInit(JNIEnv * env, jobject thisObj, jint prec, jint method,
                                                                                          jlong nComponents)
{
    return jniBatch<pca::transform::Method, Batch, defaultDense>::newObj(prec, method, nComponents);
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformBatch_cGetParameter(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                  jint prec, jint method)
{
    return jniBatch<pca::transform::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformBatch_cGetInput(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                              jint method)
{
    return jniBatch<pca::transform::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformBatch_cGetResult(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                               jint prec, jint method)
{
    return jniBatch<pca::transform::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformBatch_cSetResult(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                              jint method, jlong resultAddr)
{
    jniBatch<pca::transform::Method, Batch, defaultDense>::setResult<pca::transform::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformBatch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformBatch_cClone(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                           jint method)
{
    return jniBatch<pca::transform::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
