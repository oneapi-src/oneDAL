/* file: batch_iface.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
#include "com_intel_daal_algorithms_low_order_moments_BatchImpl.h"
#include "java_batch.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::low_order_moments;

extern "C"
{
    /*
     * Class:     com_intel_daal_algorithms_low_order_moments_BatchImpl
     * Method:    cSetResult
     * Signature: (JIIJ)V
     */
    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_BatchImpl_cSetResult(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                   jint prec, jint method, jlong resultAddr)
    {
        jniBatch<low_order_moments::Method, Batch, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR,
                 sumCSR>::setResult<low_order_moments::Result>(prec, method, algAddr, resultAddr);
    }

    /*
     * Class:     com_intel_daal_algorithms_low_order_moments_BatchImpl
     * Method:    cGetResult
     * Signature: (JII)J
     */
    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_BatchImpl_cGetResult(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                    jint prec, jint method)
    {
        return jniBatch<low_order_moments::Method, Batch, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR, sumCSR>::getResult(
            prec, method, algAddr);
    }

    /*
     * Class:     com_intel_daal_algorithms_low_order_moments_BatchImpl
     * Method:    cInitBatchImpl
     * Signature: ()J
     */
    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_BatchImpl_cInitBatchImpl(JNIEnv * env, jobject thisObj)
    {
        using namespace daal;
        using namespace daal::algorithms::low_order_moments;
        using namespace daal::services;

        JavaVM * jvm;

        // Get pointer to the Java VM interface function table
        jint status = env->GetJavaVM(&jvm);
        if (status != 0)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "Unable to get pointer to the Java VM interface function table");
            return 0;
        }
        SharedPtr<JavaBatch> * batch = new SharedPtr<JavaBatch>(new JavaBatch(jvm, thisObj));

        return (jlong)batch;
    }

    /*
     * Class:     com_intel_daal_algorithms_low_order_moments_BatchImpl
     * Method:    cDispose
     * Signature: (J)V
     */
    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_BatchImpl_cDispose(JNIEnv * env, jobject thisObj, jlong initAddr)
    {
        SharedPtr<BatchImpl> * batch = (SharedPtr<BatchImpl> *)initAddr;
        delete batch;
    }
}
