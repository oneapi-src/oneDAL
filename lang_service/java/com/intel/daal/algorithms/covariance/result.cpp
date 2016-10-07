/* file: result.cpp */
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

#include "daal.h"
#include "covariance/JResult.h"

#include "common_defines.i"
#include "covariance_types.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<covariance::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Result
 * Method:    cGetResult
 * Signature: (JIIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_Result_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode, jint step)
{
    using namespace daal::algorithms::covariance;
    using namespace daal::services;
    SerializationIfacePtr *ptr = new SerializationIfacePtr();

    if(cmode == jBatch)
    {
        return jniBatch<covariance::Method, Batch, defaultDense, singlePassDense, sumDense,
            fastCSR, singlePassCSR, sumCSR>::getResult(prec, method, algAddr);
    }
    else if(cmode == jOnline)
    {
        return jniOnline<covariance::Method, Online, defaultDense, singlePassDense, sumDense,
            fastCSR, singlePassCSR, sumCSR>::getResult(prec, method, algAddr);
    }
    else if(cmode == jDistributed)
    {
        if(step == jStep1Local)
        {
            return jniDistributed<step1Local, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
                fastCSR, singlePassCSR, sumCSR>::getResult(prec, method, algAddr);
        }
        else if(step == jStep2Master)
        {
            return jniDistributed<step2Master, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
                fastCSR, singlePassCSR, sumCSR>::getResult(prec, method, algAddr);
        }
    }

    return (jlong)ptr;
}
/*
 * Class:     com_intel_daal_algorithms_covariance_Result
 * Method:    cGetResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<covariance::Result>::get<covariance::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Result
 * Method:    cSetResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<covariance::Result>::set<covariance::ResultId, NumericTable>(resAddr, id, ntAddr);
}
