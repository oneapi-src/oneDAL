/* file: init_parameter.cpp */
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
