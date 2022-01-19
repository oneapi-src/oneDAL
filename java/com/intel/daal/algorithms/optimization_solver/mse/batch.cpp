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

#include "daal.h"

#include "com_intel_daal_algorithms_optimization_solver_mse_Batch.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_mse_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_mse_Batch_cInit(JNIEnv *, jobject, jint prec, jint method,
                                                                                            jlong numberOfTerms)
{
    return jniBatch<mse::Method, mse::Batch, mse::defaultDense>::newObj(prec, method, numberOfTerms);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_mse_Batch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_mse_Batch_cClone(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                             jint method)
{
    return jniBatch<mse::Method, mse::Batch, mse::defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_mse_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_mse_Batch_cGetInput(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                jint prec, jint method)
{
    return jniBatch<mse::Method, mse::Batch, mse::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_mse_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_mse_Batch_cGetParameter(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                    jint prec, jint method)
{
    return jniBatch<mse::Method, mse::Batch, mse::defaultDense>::getParameter(prec, method, algAddr);
}
