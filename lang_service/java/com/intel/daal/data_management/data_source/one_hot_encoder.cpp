/* file: one_hot_encoder.cpp */
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

#include "com_intel_daal_data_management_data_source_OneHotEncoder.h"

#include "csv_feature_manager.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_1source_OneHotEncoder
 * Method:    cDispose
 * Signature:(JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_OneHotEncoder_cInit(JNIEnv * env, jobject obj, jlong idx, jlong nCats)
{
    services::SharedPtr<ModifierIface> * ptr = new services::SharedPtr<ModifierIface>(new OneHotEncoder(idx, nCats));
    return (jlong)ptr;
}
