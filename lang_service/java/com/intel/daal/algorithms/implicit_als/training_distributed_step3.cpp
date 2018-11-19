/* file: training_distributed_step3.cpp */
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

#include "implicit_als/training/JDistributedStep3Local.h"

#include "implicit_als_training_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3Local
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3Local_cInit
  (JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step3Local, algorithms::implicit_als::training::Method, Distributed, fastCSR>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3Local
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3Local_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Local, algorithms::implicit_als::training::Method, Distributed, fastCSR>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3Local
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3Local_cGetPartialResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Local, algorithms::implicit_als::training::Method, Distributed, fastCSR>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3Local
 * Method:    cSetPartialResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3Local_cSetPartialResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr)
{
    jniDistributed<step3Local, algorithms::implicit_als::training::Method, Distributed, fastCSR>::
        setPartialResult<DistributedPartialResultStep3>(prec, method, algAddr, partialResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3Local
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3Local_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Local, algorithms::implicit_als::training::Method, Distributed, fastCSR>::getClone(prec, method, algAddr);
}
