/* file: feature_manager.cpp */
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

#include "com_intel_daal_data_management_data_source_FeatureManager.h"

#include "csv_feature_manager.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_1source_FeatureManager
 * Method:    cSetDelimiter
 * Signature:(JC)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_FeatureManager_cSetDelimiter(JNIEnv * env, jobject obj, jlong ptr,
                                                                                                      jchar delimiter)
{
    services::SharedPtr<CSVFeatureManager> featureManager = *((services::SharedPtr<CSVFeatureManager> *)ptr);
    featureManager->setDelimiter((char)delimiter);
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_FeatureManager
 * Method:    cAddModifier
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_FeatureManager_cAddModifier(JNIEnv * env, jobject obj, jlong ptr,
                                                                                                     jlong modifierPtr)
{
    services::SharedPtr<CSVFeatureManager> featureManager = *((services::SharedPtr<CSVFeatureManager> *)ptr);
    services::SharedPtr<ModifierIface> modifier           = *((services::SharedPtr<ModifierIface> *)modifierPtr);
    featureManager->addModifier(*modifier);
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_FeatureManager
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_FeatureManager_cDispose(JNIEnv * env, jobject obj, jlong ptr)
{
    delete (services::SharedPtr<CSVFeatureManager> *)ptr;
}
