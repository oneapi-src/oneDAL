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
#include "multi_class_classifier/JModel.h"
#include "daal.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Model
 * Method:    cGetNumberOfTwoClassClassifierModels
 * Signature:(JJ)I
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Model_cGetNumberOfTwoClassClassifierModels
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    multi_class_classifier::ModelPtr models = *(multi_class_classifier::ModelPtr *)modAddr;
    return (jlong)models->getNumberOfTwoClassClassifierModels();
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Model
 * Method:    cGetTwoClassClassifierModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Model_cGetTwoClassClassifierModel
(JNIEnv *env, jobject thisObj, jlong modAddr, jlong idx)
{
    multi_class_classifier::ModelPtr models = *(multi_class_classifier::ModelPtr *)modAddr;
    SerializationIfacePtr * dShPtr = new SerializationIfacePtr();
    *dShPtr = models->getTwoClassClassifierModel(idx);
    return (jlong)dShPtr;
}
