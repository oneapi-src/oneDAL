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
#include "distributions/bernoulli/JBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_bernoulli_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_bernoulli_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jdouble p)
{
    return jniBatch<distributions::bernoulli::Method, distributions::bernoulli::Batch, distributions::bernoulli::defaultDense>::newObj(
               prec, method, p);
}

/*
 * Class:     com_intel_daal_algorithms_distributions_bernoulli_Batch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_bernoulli_Batch_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<distributions::bernoulli::Method, distributions::bernoulli::Batch, distributions::bernoulli::defaultDense>::getParameter(
        prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_distributions_bernoulli_Batch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_bernoulli_Batch_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<distributions::bernoulli::Method, distributions::bernoulli::Batch, distributions::bernoulli::defaultDense>::getResult(
        prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_distributions_bernoulli_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_bernoulli_Batch_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<distributions::bernoulli::Method, distributions::bernoulli::Batch, distributions::bernoulli::defaultDense>::getClone(
        prec, method, algAddr);
}
