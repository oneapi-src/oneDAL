/* file: string_data_source.cpp */
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

#include "JStringDataSource.h"

#include "string_data_source.h"
#include "csv_feature_manager.h"
#include "common_helpers_functions.h"

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
        env->ReleaseStringUTFChars(jData, data);
        DAAL_CHECK_THROW(services::Status(services::ErrorMemoryAllocationFailed));
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

    if(!ds->status())
    {
        const services::Status s = ds->status();
        delete ds;
        DAAL_CHECK_THROW(s);
        return (jlong)0;
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
        env->ReleaseStringUTFChars(jData, data);
        DAAL_CHECK_THROW(services::Status(services::ErrorMemoryAllocationFailed));
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
    DAAL_CHECK_THROW(((StringDataSource<CSVFeatureManager> *)ptr)->status());
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

/*
 * Class:     com_intel_daal_data_management_data_source_StringDataSource
 * Method:    cGetFeatureManager
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_StringDataSource_cGetFeatureManager
(JNIEnv *env, jobject obj, jlong ptr)
{
    StringDataSource<CSVFeatureManager> *ds = (StringDataSource<CSVFeatureManager> *)ptr;
    services::SharedPtr<CSVFeatureManager>* featureManager =
        new services::SharedPtr<CSVFeatureManager>(&(ds->getFeatureManager()), services::EmptyDeleter());

    return (jlong)featureManager;
}
