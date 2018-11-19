/* file: batch.cpp */
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
#include "engines/mcg59/JBatch.h"
#include "engines/mcg59/JMethod.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

#define defaultDenseMethod com_intel_daal_algorithms_engines_mcg59_Method_defaultDenseId

/*
 * Class:     com_intel_daal_algorithms_engines_mcg59_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_mcg59_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jint seed)
{
    jlong addr = 0;

    if(prec == 0)
    {
        if(method == defaultDenseMethod)
        {
            SharedPtr<AlgorithmIface> *alg = new SharedPtr<AlgorithmIface>(engines::mcg59::Batch<double, engines::mcg59::defaultDense>::create(seed));
            addr = (jlong)alg;
        }
    }
    else
    {
        if(method == defaultDenseMethod)
        {
            SharedPtr<AlgorithmIface> *alg = new SharedPtr<AlgorithmIface>(engines::mcg59::Batch<float, engines::mcg59::defaultDense>::create(seed));
            addr = (jlong)alg;
        }
    }

    return addr;
}

/*
 * Class:     com_intel_daal_algorithms_engines_mcg59_Batch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_mcg59_Batch_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<engines::mcg59::Method, engines::mcg59::Batch, engines::mcg59::defaultDense>::getResult(
        prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_engines_mcg59_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_mcg59_Batch_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<engines::mcg59::Method, engines::mcg59::Batch, engines::mcg59::defaultDense>::getClone(
        prec, method, algAddr);
}
