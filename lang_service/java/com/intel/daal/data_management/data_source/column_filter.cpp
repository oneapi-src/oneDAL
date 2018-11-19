/* file: column_filter.cpp */
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
#include "JColumnFilter.h"

#include "csv_feature_manager.h"
#include "data_collection.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cInit
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cInit
(JNIEnv *env, jobject obj)
{
    services::SharedPtr<ModifierIface>* ptr = new services::SharedPtr<ModifierIface>(new ColumnFilter());
    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cOdd
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cOdd
(JNIEnv *env, jobject obj, jlong ptr)
{
    services::SharedPtr<ColumnFilter> columnFilter =
        services::staticPointerCast<ColumnFilter, ModifierIface>(
            (*(services::SharedPtr<ModifierIface> *)ptr));
    columnFilter->odd();
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cEven
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cEven
(JNIEnv *env, jobject obj, jlong ptr)
{
    services::SharedPtr<ColumnFilter> columnFilter =
        services::staticPointerCast<ColumnFilter, ModifierIface>(
            (*(services::SharedPtr<ModifierIface> *)ptr));
    columnFilter->even();
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cNone
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cNone
(JNIEnv *env, jobject obj, jlong ptr)
{
    services::SharedPtr<ColumnFilter> columnFilter =
        services::staticPointerCast<ColumnFilter, ModifierIface>(
            (*(services::SharedPtr<ModifierIface> *)ptr));
    columnFilter->none();
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_ColumnFilter
 * Method:    cList
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_ColumnFilter_cList
(JNIEnv *env, jobject obj, jlong ptr, jlongArray valid)
{
    services::SharedPtr<ColumnFilter> columnFilter =
        services::staticPointerCast<ColumnFilter, ModifierIface>(
            (*(services::SharedPtr<ModifierIface> *)ptr));
    size_t n = env->GetArrayLength(valid);
    jlong* arr = env->GetLongArrayElements(valid, 0);
    services::Collection<size_t> collection(n);
    for (int i = 0; i < n; i++)
    {
        collection[i] = (size_t)arr[i];
    }
    columnFilter->list(collection);
    env->ReleaseLongArrayElements(valid, arr, JNI_ABORT);
}
