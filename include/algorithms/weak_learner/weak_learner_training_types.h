/* file: weak_learner_training_types.h */
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
//  Implementation of the base classes used on the training stage
//  of weak learners algorithms
//--
*/

#ifndef __WEAK_LEARNER_TRAINING_TYPES_H__
#define __WEAK_LEARNER_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/weak_learner/weak_learner_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for working with weak learners
 */
namespace weak_learner
{
/**
 * @defgroup weak_learner_training Training
 * \copydoc daal::algorithms::weak_learner::training
 * @ingroup weak_learner
 * @{
 */
/**
 * \brief Contains classes for training models of the weak learners algorithms
 */
namespace training
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__WEAK_LEARNER__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with compute() method of Batch
 *        or finalizeCompute() method of Online and Distributed weak learners algorithms
 */
class Result : public daal::algorithms::classifier::training::Result
{
public:
    Result() {}
    virtual ~Result() {}

    /**
     * Returns the model trained with the weak learner  algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the weak learner  algorithm
     */
    services::SharedPtr<daal::algorithms::weak_learner::Model> get(classifier::training::ResultId id) const
    {
        return services::staticPointerCast<daal::algorithms::weak_learner::Model, data_management::SerializationIface>(Argument::get(id));
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_WEAK_LEARNER_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::classifier::training::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::Result;

} // namespace daal::algorithms::weak_learner::training
/** @} */
} // namespace daal::algorithms::weak_learner
} // namespace daal::algorithms
} // namespace daal
#endif
