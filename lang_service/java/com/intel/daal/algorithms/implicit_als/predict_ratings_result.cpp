/* file: predict_ratings_result.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <jni.h>

#include "daal.h"

#include "implicit_als/prediction/ratings/JRatingsResult.h"

#include "implicit_als_prediction_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::prediction::ratings;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsResult_cNewResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<implicit_als::prediction::ratings::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsResult
 * Method:    cGetResult
 * Signature: (JIIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsResult_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode, jint cstep)
{
    if (cmode == jBatch)
    {
        return jniBatch<implicit_als::prediction::ratings::Method, Batch, defaultDense>::
            getResult(prec, method, algAddr);
    }
    else if (cmode == jDistributed)
    {
        if (cstep == jStep1Local)
        {
            return jniDistributed<step1Local, implicit_als::prediction::ratings::Method, Distributed, defaultDense>::
                getResult(prec, method, algAddr);
        }
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsResult
 * Method:    cGetNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsResult_cGetNumericTable
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<implicit_als::prediction::ratings::Result>::get<ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsResult
 * Method:    cSetNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsResult_cSetNumericTable
  (JNIEnv *, jobject, jlong resAddr, jint id, jlong numTableAddr)
{
    jniArgument<implicit_als::prediction::ratings::Result>::set<ResultId, NumericTable>(resAddr, id, numTableAddr);
}
