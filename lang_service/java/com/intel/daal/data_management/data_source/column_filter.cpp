/* file: column_filter.cpp */
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
#include "com_intel_daal_data_management_data_source_ColumnFilter.h"

#include "csv_feature_manager.h"
#include "data_collection.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cInit
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cInit(JNIEnv * env, jobject obj)
{
    services::SharedPtr<ModifierIface> * ptr = new services::SharedPtr<ModifierIface>(new ColumnFilter());
    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cOdd
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cOdd(JNIEnv * env, jobject obj, jlong ptr)
{
    services::SharedPtr<ColumnFilter> columnFilter =
        services::staticPointerCast<ColumnFilter, ModifierIface>((*(services::SharedPtr<ModifierIface> *)ptr));
    columnFilter->odd();
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cEven
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cEven(JNIEnv * env, jobject obj, jlong ptr)
{
    services::SharedPtr<ColumnFilter> columnFilter =
        services::staticPointerCast<ColumnFilter, ModifierIface>((*(services::SharedPtr<ModifierIface> *)ptr));
    columnFilter->even();
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cNone
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cNone(JNIEnv * env, jobject obj, jlong ptr)
{
    services::SharedPtr<ColumnFilter> columnFilter =
        services::staticPointerCast<ColumnFilter, ModifierIface>((*(services::SharedPtr<ModifierIface> *)ptr));
    columnFilter->none();
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cList
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cList(JNIEnv * env, jobject obj, jlong ptr, jlongArray valid)
{
    services::SharedPtr<ColumnFilter> columnFilter =
        services::staticPointerCast<ColumnFilter, ModifierIface>((*(services::SharedPtr<ModifierIface> *)ptr));
    size_t n    = env->GetArrayLength(valid);
    jlong * arr = env->GetLongArrayElements(valid, 0);
    services::Collection<size_t> collection(n);
    for (int i = 0; i < n; i++)
    {
        collection[i] = (size_t)arr[i];
    }
    columnFilter->list(collection);
    env->ReleaseLongArrayElements(valid, arr, JNI_ABORT);
}
