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

#include "com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Batch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Batch_cInit(JNIEnv *, jobject, jint prec,
                                                                                                             jint method, jlong numberOfTerms,
                                                                                                             jlong nClasses)
{
    return jniBatch<cross_entropy_loss::Method, cross_entropy_loss::Batch, cross_entropy_loss::defaultDense>::newObj(prec, method, numberOfTerms,
                                                                                                                     nClasses);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Batch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Batch_cClone(JNIEnv * env, jobject thisObj,
                                                                                                              jlong algAddr, jint prec, jint method)
{
    return jniBatch<cross_entropy_loss::Method, cross_entropy_loss::Batch, cross_entropy_loss::defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Batch_cGetInput(JNIEnv * env, jobject thisObj,
                                                                                                                 jlong algAddr, jint prec,
                                                                                                                 jint method)
{
    return jniBatch<cross_entropy_loss::Method, cross_entropy_loss::Batch, cross_entropy_loss::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Batch_cGetParameter(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong algAddr, jint prec,
                                                                                                                     jint method)
{
    return jniBatch<cross_entropy_loss::Method, cross_entropy_loss::Batch, cross_entropy_loss::defaultDense>::getParameter(prec, method, algAddr);
}
