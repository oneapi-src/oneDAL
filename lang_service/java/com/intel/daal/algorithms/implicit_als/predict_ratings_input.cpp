/* file: predict_ratings_input.cpp */
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

#include "implicit_als/prediction/ratings/JRatingsInput.h"

#include "implicit_als_prediction_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::prediction::ratings;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsInput
 * Method:    cInit
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsInput_cInit
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<implicit_als::prediction::ratings::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsInput
 * Method:    cSetModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsInput_cSetModel
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong modelAddr)
{
    jniInput<implicit_als::prediction::ratings::Input>::set<ModelInputId, algorithms::implicit_als::Model>(inputAddr, id, modelAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsInput
 * Method:    cGetModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsInput_cGetModel
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<implicit_als::prediction::ratings::Input>::get<ModelInputId, algorithms::implicit_als::Model>(inputAddr, id);
}
