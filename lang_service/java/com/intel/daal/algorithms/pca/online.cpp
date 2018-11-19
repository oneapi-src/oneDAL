/* file: online.cpp */
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

#include "pca/JOnline.h"
#include "pca/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::pca;

/*
 * Class:     com_intel_daal_algorithms_pca_Online
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_Online_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniOnline<pca::Method, Online, correlationDense, svdDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_pca_Online
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_Online_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<pca::Method, Online, correlationDense, svdDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_Online
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_Online_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<pca::Method, Online, correlationDense, svdDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_Online
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_Online_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<pca::Method, Online, correlationDense, svdDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_Online
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_Online_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<pca::Method, Online, correlationDense, svdDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_Online
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_Online_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniOnline<pca::Method, Online, correlationDense, svdDense>::setResult<pca::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_Online
 * Method:    cSetPartialResult
 * Signature: (JIIJZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_Online_cSetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
{
    jniOnline<pca::Method, Online, correlationDense, svdDense>::
        setPartialResultImpl<pca::PartialResult>(prec, method, algAddr, partialResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_Online
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_Online_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<pca::Method, Online, correlationDense, svdDense>::getClone(prec, method, algAddr);
}
