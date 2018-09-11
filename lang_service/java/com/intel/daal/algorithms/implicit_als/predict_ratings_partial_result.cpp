/* file: predict_ratings_partial_result.cpp */
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

#include "implicit_als/prediction/ratings/JRatingsPartialResult.h"

#include "implicit_als_prediction_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::prediction::ratings;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsPartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsPartialResult_cNewPartialResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<implicit_als::prediction::ratings::PartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsPartialResult
 * Method:    cSetResult
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsPartialResult_cSetResult
  (JNIEnv *, jobject, jlong presAddr, jint id, jlong resAddr)
{
    jniArgument<implicit_als::prediction::ratings::PartialResult>::
        set<PartialResultId, implicit_als::prediction::ratings::Result>(presAddr, id, resAddr);
}
