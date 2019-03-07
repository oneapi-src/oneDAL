/* file: batch.cpp */
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
#include "engines/mt2203/JBatch.h"
#include "engines/mt2203/JMethod.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

#define defaultDenseMethod com_intel_daal_algorithms_engines_mt2203_Method_defaultDenseId

/*
 * Class:     com_intel_daal_algorithms_engines_mt2203_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_mt2203_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jint seed)
{
    jlong addr = 0;

    if(prec == 0)
    {
        if(method == defaultDenseMethod)
        {
            SharedPtr<AlgorithmIface> *alg = new SharedPtr<AlgorithmIface>(engines::mt2203::Batch<double, engines::mt2203::defaultDense>::create(seed));
            addr = (jlong)alg;
        }
    }
    else
    {
        if(method == defaultDenseMethod)
        {
            SharedPtr<AlgorithmIface> *alg = new SharedPtr<AlgorithmIface>(engines::mt2203::Batch<float, engines::mt2203::defaultDense>::create(seed));
            addr = (jlong)alg;
        }
    }

    return addr;
}

/*
 * Class:     com_intel_daal_algorithms_engines_mt2203_Batch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_mt2203_Batch_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<engines::mt2203::Method, engines::mt2203::Batch, engines::mt2203::defaultDense>::getResult(
        prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_engines_mt2203_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_mt2203_Batch_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<engines::mt2203::Method, engines::mt2203::Batch, engines::mt2203::defaultDense>::getClone(
        prec, method, algAddr);
}
