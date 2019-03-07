/* file: common_defines.i */
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

#include "JComputeMode.h"
#include "JComputeStep.h"

#define jBatch          com_intel_daal_algorithms_ComputeMode_batchValue
#define jOnline         com_intel_daal_algorithms_ComputeMode_onlineValue
#define jDistributed    com_intel_daal_algorithms_ComputeMode_distributedValue

#define jStep1Local     com_intel_daal_algorithms_ComputeStep_step1LocalValue
#define jStep2Master    com_intel_daal_algorithms_ComputeStep_step2MasterValue
#define jStep3Local     com_intel_daal_algorithms_ComputeStep_step3LocalValue


namespace daal
{

const int SERIALIZATION_JAVANIO_CSR_NT_ID                                                       = 9000;
const int SERIALIZATION_JAVANIO_HOMOGEN_NT_ID                                                   = 10010;
const int SERIALIZATION_JAVANIO_AOS_NT_ID                                                       = 10020;
const int SERIALIZATION_JAVANIO_SOA_NT_ID                                                       = 10030;
const int SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID                                           = 10040;
const int SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID                                          = 10050;
const int SERIALIZATION_JAVANIO_HOMOGEN_TENSOR_ID                                               = 21000;

} // namespace daal


#define IMPLEMENT_SERIALIZABLE_TAG(Class,Tag) \
    int Class<Tag>::serializationTag() { return Tag; } \
    int Class<Tag>::getSerializationTag() const { return Class<Tag>::serializationTag(); }

#define IMPLEMENT_SERIALIZABLE_TAGT(Class,Tag) \
    template<> int Class<Tag>::serializationTag() { return Tag; } \
    template<> int Class<Tag>::getSerializationTag() const { return Class<Tag>::serializationTag(); }
