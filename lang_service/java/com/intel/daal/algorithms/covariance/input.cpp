/* file: input.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_covariance_Offline */

#include "daal.h"
#include "covariance/JInput.h"

#include "common_defines.i"
#include "covariance_types.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

/*
 * Class:     com_intel_daal_algorithms_covariance_Input
 * Method:    cInit
 * Signature: (JIIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_Input_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode, jint step)
{
    if(cmode == jBatch)
    {
        return jniBatch<covariance::Method, Batch, defaultDense, singlePassDense, sumDense,
            fastCSR, singlePassCSR, sumCSR>::getInput(prec, method, algAddr);
    }
    else if(cmode == jOnline)
    {
        return jniOnline<covariance::Method, Online, defaultDense, singlePassDense, sumDense,
            fastCSR, singlePassCSR, sumCSR>::getInput(prec, method, algAddr);
    }
    else if(cmode == jDistributed)
    {
        if(step == jStep1Local)
        {
            return jniDistributed<step1Local, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
                fastCSR, singlePassCSR, sumCSR>::getInput(prec, method, algAddr);
        }
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Input
 * Method:    cSetCInputObject
 * Signature: (JJIIII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_Input_cSetCInputObjectBatch
(JNIEnv *env, jobject thisObj, jlong inputAddr, jlong algAddr, jint prec, jint method)
// somehow this function isn't called if has >4 parameters
{
    jniBatch<covariance::Method, Batch, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::setInput<covariance::Input>(prec, method, algAddr, inputAddr);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Input
 * Method:    cSetCInputObject
 * Signature: (JJIIII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_Input_cSetCInputObjectOnline
(JNIEnv *env, jobject thisObj, jlong inputAddr, jlong algAddr, jint prec, jint method)
// somehow this function isn't called if has >4 parameters
{
    jniOnline<covariance::Method, Online, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::setInput<covariance::Input>(prec, method, algAddr, inputAddr);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Input
 * Method:    cSetCInputObject
 * Signature: (JJIIII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_Input_cSetCInputObjectDistributedStep1Local
(JNIEnv *env, jobject thisObj, jlong inputAddr, jlong algAddr, jint prec, jint method)
// somehow this function isn't called if has >4 parameters
{
    jniDistributed<step1Local, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::setInput<covariance::Input>(prec, method, algAddr, inputAddr);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Input
 * Method:    cSetInput
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_Input_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<covariance::Input>::set<covariance::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Input
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_Input_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<covariance::Input>::get<covariance::InputId, NumericTable>(inputAddr, id);
}
