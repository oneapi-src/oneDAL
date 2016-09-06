/* file: batch_iface.cpp */
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
#include "covariance/JBatchIface.h"
#include "common_defines.i"
#include "covariance_types.i"
#include "java_batch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

extern "C"
{

    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_BatchIface_cSetResult
    (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
    {
        jniBatch<covariance::Method, Batch, defaultDense, singlePassDense, sumDense,
            fastCSR, singlePassCSR, sumCSR>::setResult<covariance::Result>(prec, method, algAddr, resultAddr);
    }

    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_BatchIface_cGetResult
    (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
    {
        return jniBatch<covariance::Method, Batch, defaultDense, singlePassDense, sumDense,
            fastCSR, singlePassCSR, sumCSR>::getResult(prec, method, algAddr);
    }

    /*
     * Class:     com_intel_daal_algorithms_covariance_BatchIface
     * Method:    cInitBatchIface
     * Signature: ()J
     */
    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_BatchIface_cInitBatchIface
    (JNIEnv *env, jobject thisObj)
    {
        using namespace daal;
        using namespace daal::algorithms::covariance;
        using namespace daal::services;

        JavaVM *jvm;


        // Get pointer to the Java VM interface function table
        jint status = env->GetJavaVM(&jvm);
        if(status != 0)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "Unable to get pointer to the Java VM interface function table");
            return 0;
        }
        SharedPtr<JavaBatch> *covBatch = new SharedPtr<JavaBatch>(new JavaBatch(jvm, thisObj));

        return (jlong)covBatch;
    }

    /*
     * Class:     com_intel_daal_algorithms_covariance_BatchIface
     * Method:    cDispose
     * Signature: (J)V
     */
    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_BatchIface_cDispose
    (JNIEnv *env, jobject thisObj, jlong initAddr)
    {
        SharedPtr<BatchIface> *covBatch = (SharedPtr<BatchIface> *)initAddr;
        delete covBatch;
    }
}
