/* file: batch_iface.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include <jni.h>

#include "daal.h"
#include "covariance/JBatchImpl.h"
#include "common_defines.i"
#include "covariance_types.i"
#include "java_batch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

extern "C"
{
    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_BatchImpl_cGetResult
    (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
    {
        return jniBatch<covariance::Method, Batch, defaultDense, singlePassDense, sumDense,
            fastCSR, singlePassCSR, sumCSR>::getResult(prec, method, algAddr);
    }

    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_BatchImpl_cSetResult
    (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
    {
        jniBatch<covariance::Method, Batch, defaultDense, singlePassDense, sumDense,
            fastCSR, singlePassCSR, sumCSR>::setResult<covariance::Result>(prec, method, algAddr, resultAddr);
    }

    /*
     * Class:     com_intel_daal_algorithms_covariance_BatchImpl
     * Method:    cInitBatchImpl
     * Signature: ()J
     */
    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_BatchImpl_cInitBatchImpl
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
     * Class:     com_intel_daal_algorithms_covariance_BatchImpl
     * Method:    cDispose
     * Signature: (J)V
     */
    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_BatchImpl_cDispose
    (JNIEnv *env, jobject thisObj, jlong initAddr)
    {
        SharedPtr<BatchImpl> *covBatch = (SharedPtr<BatchImpl> *)initAddr;
        delete covBatch;
    }
}
