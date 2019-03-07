/* file: partial_model.cpp */
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

#include <jni.h>

#include "daal.h"

#include "implicit_als/JPartialModel.h"

using namespace daal;
using namespace daal::algorithms::implicit_als;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_PartialModel
 * Method:    cNewPartialModel
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_PartialModel_cNewPartialModel
  (JNIEnv *env, jobject thisObj, jlong factorsAddr, jlong indicesAddr)
{
    SerializationIfacePtr *factorsShPtr = (SerializationIfacePtr *)factorsAddr;
    SerializationIfacePtr *indicesShPtr = (SerializationIfacePtr *)indicesAddr;
    NumericTablePtr factors =
        services::staticPointerCast<NumericTable, SerializationIface>(*factorsShPtr);
    NumericTablePtr indices =
        services::staticPointerCast<NumericTable, SerializationIface>(*indicesShPtr);
    return (jlong)(new SerializationIfacePtr(new PartialModel(factors, indices)));
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_PartialModel
 * Method:    cGetFactors
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_PartialModel_cGetFactors
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    SerializationIfacePtr *factors = new SerializationIfacePtr();
    PartialModel *pModel = static_cast<PartialModel *>(((SerializationIfacePtr *)modAddr)->get());

    *factors = pModel->getFactors();

    return (jlong)factors;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_PartialModel
 * Method:    cGetIndices
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_PartialModel_cGetIndices
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    SerializationIfacePtr *indices = new SerializationIfacePtr();
    PartialModel *pModel = static_cast<PartialModel *>(((SerializationIfacePtr *)modAddr)->get());

    *indices = pModel->getIndices();

    return (jlong)indices;
}
