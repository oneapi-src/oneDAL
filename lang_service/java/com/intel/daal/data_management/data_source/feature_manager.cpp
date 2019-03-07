/* file: feature_manager.cpp */
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

#include "JFeatureManager.h"

#include "csv_feature_manager.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_1source_FeatureManager
 * Method:    cSetDelimiter
 * Signature:(JC)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_FeatureManager_cSetDelimiter
(JNIEnv *env, jobject obj, jlong ptr, jchar delimiter)
{
    services::SharedPtr<CSVFeatureManager> featureManager = *((services::SharedPtr<CSVFeatureManager> *)ptr);
    featureManager->setDelimiter((char)delimiter);
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_FeatureManager
 * Method:    cAddModifier
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_FeatureManager_cAddModifier
(JNIEnv *env, jobject obj, jlong ptr, jlong modifierPtr)
{
    services::SharedPtr<CSVFeatureManager> featureManager = *((services::SharedPtr<CSVFeatureManager> *)ptr);
    services::SharedPtr<ModifierIface> modifier = *((services::SharedPtr<ModifierIface> *)modifierPtr);
    featureManager->addModifier(*modifier);
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_FeatureManager
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_FeatureManager_cDispose
(JNIEnv *env, jobject obj, jlong ptr)
{
    delete(services::SharedPtr<CSVFeatureManager> *)ptr;
}
