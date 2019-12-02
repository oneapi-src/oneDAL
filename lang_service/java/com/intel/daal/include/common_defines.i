/* file: common_defines.i */
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

#include "com_intel_daal_algorithms_ComputeMode.h"
#define jBatch       com_intel_daal_algorithms_ComputeMode_batchValue
#define jOnline      com_intel_daal_algorithms_ComputeMode_onlineValue
#define jDistributed com_intel_daal_algorithms_ComputeMode_distributedValue

#include "com_intel_daal_algorithms_ComputeStep.h"
#define jStep1Local  com_intel_daal_algorithms_ComputeStep_step1LocalValue
#define jStep2Master com_intel_daal_algorithms_ComputeStep_step2MasterValue
#define jStep3Local  com_intel_daal_algorithms_ComputeStep_step3LocalValue

namespace daal
{
const int SERIALIZATION_JAVANIO_CSR_NT_ID              = 9000;
const int SERIALIZATION_JAVANIO_HOMOGEN_NT_ID          = 10010;
const int SERIALIZATION_JAVANIO_AOS_NT_ID              = 10020;
const int SERIALIZATION_JAVANIO_SOA_NT_ID              = 10030;
const int SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID  = 10040;
const int SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID = 10050;

} // namespace daal

#define IMPLEMENT_SERIALIZABLE_TAG(Class, Tag)         \
    int Class<Tag>::serializationTag() { return Tag; } \
    int Class<Tag>::getSerializationTag() const { return Class<Tag>::serializationTag(); }

#define IMPLEMENT_SERIALIZABLE_TAGT(Class, Tag) \
    template <>                                 \
    int Class<Tag>::serializationTag()          \
    {                                           \
        return Tag;                             \
    }                                           \
    template <>                                 \
    int Class<Tag>::getSerializationTag() const \
    {                                           \
        return Class<Tag>::serializationTag();  \
    }
