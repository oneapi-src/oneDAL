/* file: make_categorical.cpp */
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

#include "JMakeCategorical.h"

#include "csv_feature_manager.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_1source_MakeCategorical
 * Method:    cDispose
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_MakeCategorical_cInit
(JNIEnv *env, jobject obj, jlong idx)
{
    services::SharedPtr<ModifierIface>* ptr = new services::SharedPtr<ModifierIface>(new MakeCategorical(idx));
    return (jlong)ptr;
}
