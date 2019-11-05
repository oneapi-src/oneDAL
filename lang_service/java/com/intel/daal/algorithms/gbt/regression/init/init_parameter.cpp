/* file: init_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "com_intel_daal_algorithms_gbt_regression_init_InitParameter.h"

using namespace daal::algorithms::gbt::regression::init;

/*
 * Class:     com_intel_daal_algorithms_gbt_regression_init_Parameter
 * Method:    cGetMaxBins
 * Signature:(J)D
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitParameter_cGetMaxBins
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->maxBins;
}

/*
 * Class:     com_intel_daal_algorithms_gbt_regression_init_Parameter
 * Method:    cGetMinBinSize
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitParameter_cGetMinBinSize
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->minBinSize;
}

/*
 * Class:     com_intel_daal_algorithms_gbt_regression_init_Parameter
 * Method:    cSetMaxBins
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitParameter_cSetMaxBins
(JNIEnv *, jobject, jlong parameterAddress, jlong maxBins)
{
    ((Parameter *)parameterAddress)->maxBins = maxBins;
}

/*
 * Class:     com_intel_daal_algorithms_gbt_regression_init_Parameter
 * Method:    cSetMinBinSize
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitParameter_cSetMinBinSize
(JNIEnv *, jobject, jlong parameterAddress, jlong minBinSize)
{
    ((Parameter *)parameterAddress)->minBinSize = minBinSize;
}
