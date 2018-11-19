/* file: input.cpp */
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
