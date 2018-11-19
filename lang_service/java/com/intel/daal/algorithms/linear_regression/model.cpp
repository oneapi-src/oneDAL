/* file: model.cpp */
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
#include "linear_regression/JModel.h"
#include "daal.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Model
 * Method:    cGetBeta
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_Model_cGetBeta
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    algorithms::linear_regression::ModelPtr res = *(algorithms::linear_regression::ModelPtr *)modAddr;

    *nt = res->getBeta();

    return (jlong)nt;
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Model
 * Method:    cGetNumberOfFeatures
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_Model_cGetNumberOfFeatures
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    algorithms::linear_regression::ModelPtr res = *(algorithms::linear_regression::ModelPtr *)modAddr;

    return (jlong)res->getNumberOfFeatures();
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Model
 * Method:    cGetNumberOfBetas
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_Model_cGetNumberOfBetas
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    algorithms::linear_regression::ModelPtr res = *(algorithms::linear_regression::ModelPtr *)modAddr;

    return (jlong)res->getNumberOfBetas();
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Model
 * Method:    cGetNumberOfResponses
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_Model_cGetNumberOfResponses
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    algorithms::linear_regression::ModelPtr res = *(algorithms::linear_regression::ModelPtr *)modAddr;

    return (jlong)res->getNumberOfResponses();
}
