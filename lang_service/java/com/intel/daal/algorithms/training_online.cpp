/* file: training_online.cpp */
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
#include "JTrainingOnline.h"

#include "daal_defines.h"
#include "algorithm.h"

using namespace daal;
using namespace daal::services;
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_TrainingOnline
 * Method:    cCompute
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingOnline_cCompute
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Training<online> > alg =
        staticPointerCast<Training<online>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    alg->compute();
    if(alg->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), (*(daal::algorithms::Training<daal::online> *)
                                                              algAddr).getErrors()->getDescription());
    }
}


/*
 * Class:     com_intel_daal_algorithms_TrainingOnline
 * Method:    cFinalizeCompute
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingOnline_cFinalizeCompute
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Training<online> > alg =
        staticPointerCast<Training<online>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    alg->finalizeCompute();
    if(alg->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), alg->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_algorithms_TrainingOnline
 * Method:    cCheckComputeParams
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingOnline_cCheckComputeParams
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Training<online> > alg =
        staticPointerCast<Training<online>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    alg->checkComputeParams();
    if(alg->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), alg->getErrors()->getDescription());
    }
}


/*
 * Class:     com_intel_daal_algorithms_TrainingOnline
 * Method:    cCheckFinalizeComputeParams
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingOnline_cCheckFinalizeComputeParams
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Training<online> > alg =
        staticPointerCast<Training<online>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    alg->checkFinalizeComputeParams();
    if(alg->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), alg->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_algorithms_TrainingOnline
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingOnline_cDispose
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    delete(SharedPtr<AlgorithmIface> *)algAddr;
}
