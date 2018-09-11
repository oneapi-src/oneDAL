/* file: online_iface.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
#include "covariance/JOnlineImpl.h"
#include "common_defines.i"
#include "covariance_types.i"
#include "java_online.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

extern "C"
{
    /*
     * Class:     com_intel_daal_algorithms_covariance_Online
     * Method:    cGetResult
     * Signature: (JII)J
     */
    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_OnlineImpl_cGetResult
    (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
    {
        return jniOnline<covariance::Method, Online, defaultDense, singlePassDense, sumDense,
                  fastCSR, singlePassCSR, sumCSR>::getResult(prec, method, algAddr);
    }

    /*
     * Class:     com_intel_daal_algorithms_covariance_Online
     * Method:    cSetResult
     * Signature: (JIIJ)V
     */
    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_OnlineImpl_cSetResult
    (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
    {
        jniOnline<covariance::Method, Online, defaultDense, singlePassDense, sumDense,
                  fastCSR, singlePassCSR, sumCSR>::setResult<covariance::Result>(prec, method, algAddr, resultAddr);
    }

    /*
     * Class:     com_intel_daal_algorithms_covariance_Online
     * Method:    cGetPartialResult
     * Signature: (JII)J
     */
    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_OnlineImpl_cGetPartialResult
    (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
    {
        return jniOnline<covariance::Method, Online, defaultDense, singlePassDense, sumDense,
                  fastCSR, singlePassCSR, sumCSR>::getPartialResult(prec, method, algAddr);
    }

    /*
     * Class:     com_intel_daal_algorithms_covariance_Online
     * Method:    cSetPartialResult
     * Signature: (JIIJZ)V
     */
    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_OnlineImpl_cSetPartialResult
    (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
    {
        jniOnline<covariance::Method, Online, defaultDense, singlePassDense, sumDense,
                  fastCSR, singlePassCSR, sumCSR>::setPartialResult<covariance::PartialResult>(prec, method, algAddr, partialResultAddr, initFlag);
    }

    /*
     * Class:     com_intel_daal_algorithms_covariance_OnlineImpl
     * Method:    cInitOnlineImpl
     * Signature: ()J
     */
    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_OnlineImpl_cInitOnlineImpl
    (JNIEnv *env, jobject thisObj)
    {
        JavaVM *jvm;

        // Get pointer to the Java VM interface function table
        jint status = env->GetJavaVM(&jvm);
        if(status != 0)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "Unable to get pointer to the Java VM interface function table");
            return 0;
        }
        SharedPtr<JavaOnline> *covOnline = new SharedPtr<JavaOnline>(new JavaOnline(jvm, thisObj));
        return (jlong)covOnline;
    }

    /*
     * Class:     com_intel_daal_algorithms_covariance_OnlineImpl
     * Method:    cDispose
     * Signature: (J)V
     */
    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_OnlineImpl_cDispose
    (JNIEnv *env, jobject thisObj, jlong initAddr)
    {
        SharedPtr<OnlineImpl> *covOnline = (SharedPtr<OnlineImpl> *)initAddr;
        delete covOnline;
    }
}
