/* file: model.h */
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

/*
//++
//  Data model classes declarations
//--
*/

#ifndef __MODEL_H__
#define __MODEL_H__

#include "data_management/data/data_archive.h"
#include "services/base.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @addtogroup base_algorithms
 * @{
 */
struct ValidationMetricIface {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MODEL"></a>
 * \brief The base class for the classes that represent the models, such as linear_regression::Model
 *        or svm::Model
 * \note  The current version of the library does not support generation of validation metrics
 *        for the models
 */
class Model : public data_management::SerializationIface
{
public:
    /** Default constructor */
    Model()
    {}

    virtual ~Model() {}

    int getSerializationTag() DAAL_C11_OVERRIDE  { return 0; }
    /**
     *  Serializes the object
     *  \param[in]  archive  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Deserializes the object
     *  \param[in]  archive  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {}
};
/** @} */
} // namespace interface1
using interface1::ValidationMetricIface;
using interface1::Model;

}
} // namespace daal
#endif
