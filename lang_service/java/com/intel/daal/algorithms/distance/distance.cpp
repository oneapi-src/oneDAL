/* file: distance.cpp */
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
#include "JDistance.h"
#include "daal.h"

/*
 * Class:     com_intel_daal_algorithms_distance_Distance
 * Method:    dInitCorFast
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distance_Distance_dInitCorFast
(JNIEnv *env, jobject thisObj)
{
    using namespace daal;
    return(jlong)(new Distance<CorDistance, DistanceFast, double>());
}

/*
 * Class:     com_intel_daal_algorithms_distance_Distance
 * Method:    sInitCorFast
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distance_Distance_sInitCorFast
(JNIEnv *env, jobject thisObj)
{
    using namespace daal;
    return(jlong)(new Distance<CorDistance, DistanceFast, float>());
}

/*
 * Class:     com_intel_daal_algorithms_distance_Distance
 * Method:    dInitCosFast
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distance_Distance_dInitCosFast
(JNIEnv *env, jobject thisObj)
{
    using namespace daal;
    return(jlong)(new Distance<CosDistance, DistanceFast, double>());
}

/*
 * Class:     com_intel_daal_algorithms_distance_Distance
 * Method:    sInitCosFast
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distance_Distance_sInitCosFast
(JNIEnv *env, jobject thisObj)
{
    using namespace daal;
    return(jlong)(new Distance<CosDistance, DistanceFast, float>());
}
