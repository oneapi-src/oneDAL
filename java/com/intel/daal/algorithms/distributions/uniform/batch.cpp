/* file: batch.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
#include "com_intel_daal_algorithms_distributions_uniform_Batch.h"

#include "daal.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_uniform_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_uniform_Batch_cInit(JNIEnv * env, jobject thisObj, jint prec, jint method,
                                                                                         jdouble a, jdouble b)
{
    return jniBatch<distributions::uniform::Method, distributions::uniform::Batch, distributions::uniform::defaultDense>::newObj(prec, method, a, b);
}

/*
 * Class:     com_intel_daal_algorithms_distributions_uniform_Batch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_uniform_Batch_cInitParameter(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                  jint prec, jint method)
{
    return jniBatch<distributions::uniform::Method, distributions::uniform::Batch, distributions::uniform::defaultDense>::getParameter(prec, method,
                                                                                                                                       algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_distributions_uniform_Batch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_uniform_Batch_cGetResult(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                              jint method)
{
    return jniBatch<distributions::uniform::Method, distributions::uniform::Batch, distributions::uniform::defaultDense>::getResult(prec, method,
                                                                                                                                    algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_distributions_uniform_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_uniform_Batch_cClone(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                          jint method)
{
    return jniBatch<distributions::uniform::Method, distributions::uniform::Batch, distributions::uniform::defaultDense>::getClone(prec, method,
                                                                                                                                   algAddr);
}
