/* file: training_init_distributed_parameter.cpp */
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

#include "daal.h"

#include "implicit_als/training/init/JInitDistributedParameter.h"

using namespace daal;
using namespace daal::algorithms::implicit_als::training::init;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cSetPartition
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedParameter_cSetPartition
(JNIEnv *, jobject, jlong parAddr, jlong ntAddr)
{
    NumericTablePtr nt = NumericTable::cast(*(SerializationIfacePtr *)ntAddr);
    ((DistributedParameter *)parAddr)->partition = nt;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cGetPartition
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL ava_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedParameter_cGetPartition
(JNIEnv *, jobject, jlong parAddr)
{
    return (jlong)(new SerializationIfacePtr(((DistributedParameter *)parAddr)->partition));
}
