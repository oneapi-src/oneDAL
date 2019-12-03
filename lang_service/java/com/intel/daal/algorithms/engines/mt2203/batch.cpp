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
#include "com_intel_daal_algorithms_engines_mt2203_Batch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

#include "com_intel_daal_algorithms_engines_mt2203_Method.h"
#define defaultDenseMethod com_intel_daal_algorithms_engines_mt2203_Method_defaultDenseId

/*
 * Class:     com_intel_daal_algorithms_engines_mt2203_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_mt2203_Batch_cInit(JNIEnv * env, jobject thisObj, jint prec, jint method, jint seed)
{
    jlong addr = 0;

    if (prec == 0)
    {
        if (method == defaultDenseMethod)
        {
            SharedPtr<AlgorithmIface> * alg =
                new SharedPtr<AlgorithmIface>(engines::mt2203::Batch<double, engines::mt2203::defaultDense>::create(seed));
            addr = (jlong)alg;
        }
    }
    else
    {
        if (method == defaultDenseMethod)
        {
            SharedPtr<AlgorithmIface> * alg =
                new SharedPtr<AlgorithmIface>(engines::mt2203::Batch<float, engines::mt2203::defaultDense>::create(seed));
            addr = (jlong)alg;
        }
    }

    return addr;
}

/*
 * Class:     com_intel_daal_algorithms_engines_mt2203_Batch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_mt2203_Batch_cGetResult(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                       jint method)
{
    return jniBatch<engines::mt2203::Method, engines::mt2203::Batch, engines::mt2203::defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_engines_mt2203_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_mt2203_Batch_cClone(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                   jint method)
{
    return jniBatch<engines::mt2203::Method, engines::mt2203::Batch, engines::mt2203::defaultDense>::getClone(prec, method, algAddr);
}
