/* file: string_data_source.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "JStringDataSource.h"

#include "string_data_source.h"
#include "csv_feature_manager.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::services;

/*
 * Class:     com_intel_daal_data_1management_data_1source_StringDataSource
 * Method:    cInit
 * Signature:(Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_StringDataSource_cInit
(JNIEnv *env, jobject obj, jstring jData, jlong n)
{
    const char *data = env->GetStringUTFChars(jData, NULL);

    char *inner_data = (char *)daal_malloc( n + 1 );
    if(!inner_data)
    {
        Error e(services::ErrorMemoryAllocationFailed);
        const char *description = e.description();
        env->ThrowNew(env->FindClass("java/lang/Exception"),description);
        return (jlong)0;
    }

    for( size_t i = 0; i < n; i++ )
    {
        inner_data[i] = data[i];
    }
    inner_data[n] = '\0';

    env->ReleaseStringUTFChars(jData, data);

    DataSource *ds = new StringDataSource<CSVFeatureManager>((byte *)inner_data, DataSource::doAllocateNumericTable,
                                                             DataSource::doDictionaryFromContext);

    if(((DataSource *)ds)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), ((DataSource *)ds)->getErrors()->getDescription());
    }

    return(jlong)(ds);
}

JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_StringDataSource_cSetData
(JNIEnv *env, jobject obj, jlong ptr, jstring jData, jlong n)
{
    const char *data = env->GetStringUTFChars(jData, NULL);

    char *inner_data = (char *)daal_malloc( n + 1 );
    if(!inner_data)
    {
        Error e(services::ErrorMemoryAllocationFailed);
        const char *description = e.description();
        env->ThrowNew(env->FindClass("java/lang/Exception"),description);
        return;
    }

    for( size_t i = 0; i < n; i++ )
    {
        inner_data[i] = data[i];
    }
    inner_data[n] = '\0';

    env->ReleaseStringUTFChars(jData, data);

    ((StringDataSource<CSVFeatureManager> *)ptr)->resetData();
    ((StringDataSource<CSVFeatureManager> *)ptr)->setData((byte *)inner_data);

    if(((StringDataSource<CSVFeatureManager> *)ptr)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((StringDataSource<CSVFeatureManager> *)ptr)->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_1management_data_1source_DataSource
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_StringDataSource_cDispose
(JNIEnv *env, jobject obj, jlong ptr)
{
    const byte *data = ((StringDataSource<CSVFeatureManager> *)ptr)->getData();
    delete(DataSource *)ptr;
    if( data )
    {
        daal_free((void *)data);
    }
}
