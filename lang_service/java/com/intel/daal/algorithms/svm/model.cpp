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
#include "svm/JModel.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::svm;

/*
 * Class:     com_intel_daal_algorithms_svm_Model
 * Method:    cGetSupportVectors
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_Model_cGetSupportVectors
(JNIEnv *env, jobject obj, jlong modelAddr)
{
    NumericTablePtr *ptr = new NumericTablePtr();
    *ptr = (*(ModelPtr *)modelAddr)->getSupportVectors();

    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Model
 * Method:    cGetClassificationCoefficients
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_Model_cGetClassificationCoefficients
(JNIEnv *env, jobject obj, jlong modelAddr)
{
    NumericTablePtr *ptr = new NumericTablePtr();
    *ptr = (*(ModelPtr *)modelAddr)->getClassificationCoefficients();

    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Model
 * Method:    cGetBias
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_svm_Model_cGetBias
(JNIEnv *env, jobject obj, jlong modelAddr)
{
    return (jdouble)((*(ModelPtr *)modelAddr)->getBias());
}
