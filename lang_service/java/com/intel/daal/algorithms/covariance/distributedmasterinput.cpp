/* file: distributedmasterinput.cpp */
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
#include "covariance/JDistributedStep2MasterInput.h"

#include "covariance_types.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

/*
 * Class:     com_intel_daal_algorithms_covariance_DistributedStep2MasterInput
 * Method:    cInit
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_DistributedStep2MasterInput_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_DistributedStep2MasterInput
 * Method:    cAddInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_DistributedStep2MasterInput_cAddInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong partialResultAddr)
{
    jniInput<covariance::DistributedInput<step2Master> >::add<covariance::MasterInputId, covariance::PartialResult>(inputAddr, id, partialResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Input
 * Method:    cSetCInputObject
 * Signature: (JJIIII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_DistributedStep2MasterInput_cSetCInputObject
(JNIEnv *env, jobject thisObj, jlong inputAddr, jlong algAddr, jint prec, jint method)
// somehow this function isn't called if has >4 parameters
{
    jniDistributed<step2Master, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::setInput<covariance::DistributedInput<step2Master> >(prec, method, algAddr, inputAddr);
}
