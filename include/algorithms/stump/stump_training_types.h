/* file: stump_training_types.h */
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
//  Implementation of the interface of the decision stump training algorithm.
//--
*/

#ifndef __STUMP_TRAINING_TYPES_H__
#define __STUMP_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/weak_learner/weak_learner_training_types.h"
#include "algorithms/stump/stump_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes to work with the decision stump training algorithm
 */
namespace stump
{
/**
 * @defgroup stump_training Training
 * \copydoc daal::algorithms::stump::training
 * @ingroup stump
 * @{
 */
/**
 * \brief Contains classes to train the decision stump model
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__STUMP__TRAINING__METHOD"></a>
 * Available methods to train the decision stump model
 */
enum Method
{
    defaultDense = 0        /*!< Default method */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the decision stump training algorithm
 * in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::weak_learner::training::Result
{
public:
    Result();

    virtual ~Result() {}

    /**
     * Returns the model trained with the Stump algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the Stump algorithm
     */
    services::SharedPtr<daal::algorithms::stump::Model> get(classifier::training::ResultId id) const;

    /**
     * Allocates memory to store final results of the decision stump training algorithm
     * \tparam algorithmFPType  Data type to store prediction results
     * \param[in] input         %Input objects for the decision stump training algorithm
     * \param[in] parameter     Parameters of the decision stump training algorithm
     * \param[in] method        Decision stump training method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Check the correctness of the Result object
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameters structure
     * \param[in] method    Algorithm computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_STUMP_TRAINING_RESULT_ID; }

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
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::Result;

} // namespace daal::algorithms::stump::training
/** @} */
}
}
} // namespace daal
#endif // __STUMP_TRAINING_TYPES_H__
