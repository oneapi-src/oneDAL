/* file: file_data_source.cpp */
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

#include "JFileDataSource.h"

#include "file_data_source.h"
#include "csv_feature_manager.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_management_data_source_FileDataSource
 * Method:    cInit
 * Signature:(Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_FileDataSource_cInit
(JNIEnv *env, jobject obj, jstring jFileName)
{
    const char *fileName = env->GetStringUTFChars(jFileName, NULL);

    DataSource *ds = new FileDataSource<CSVFeatureManager>(fileName);
    env->ReleaseStringUTFChars(jFileName, fileName);
    if(!ds->status())
    {
        const services::Status s = ds->status();
        delete ds;
        DAAL_CHECK_THROW(s);
        return (jlong)0;
    }

    return(jlong)(ds);
}

/*
 * Class:     com_intel_daal_data_management_data_source_FileDataSource
 * Method:    cGetFeatureManager
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_FileDataSource_cGetFeatureManager
(JNIEnv *env, jobject obj, jlong ptr)
{
    FileDataSource<CSVFeatureManager> *ds = (FileDataSource<CSVFeatureManager> *)ptr;
    services::SharedPtr<CSVFeatureManager>* featureManager =
        new services::SharedPtr<CSVFeatureManager>(&(ds->getFeatureManager()), services::EmptyDeleter());

    return (jlong)featureManager;
}
