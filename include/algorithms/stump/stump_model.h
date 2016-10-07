/* file: stump_model.h */
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
//  Implementation of the class defining the decision stump model.
//--
*/

#ifndef __STUMP_MODEL_H__
#define __STUMP_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/weak_learner/weak_learner_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup stump Stump
 * \copydoc daal::algorithms::stump
 * @ingroup weak_learner
 * @{
 */
namespace stump
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__MODEL"></a>
 * \brief Model of the classifier trained by the stump::training::Batch algorithm.
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class Model : public weak_learner::Model
{
public:
    DAAL_DOWN_CAST_OPERATOR(Model,classifier::Model)

    size_t      splitFeature;           /*!< Index of the feature over which the split is made */
    data_management::NumericTablePtr values;     /*!< Table that contains 3 values:\n
                                                Value of the splitFeature feature that defines the split,\n
                                                Average of the weighted responses for the "left" subset,\n
                                                Average of the weighted responses for the "right" subset */

    /**
     * Constructs the decision stump model
     * \tparam modelFPType  Data type to store decision stump model data, double or float
     * \param[in] dummy     Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    Model(modelFPType dummy) :
        weak_learner::Model(),
        values(new data_management::HomogenNumericTable<modelFPType>(3, 1, data_management::NumericTable::doAllocate))
    {}

    /**
     * Empty constructor for deserialization
     */
    Model() : weak_learner::Model(), splitFeature(0), values()
    {}

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_STUMP_MODEL_ID; }
    /**
     *  Serializes the model object
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Deserializes the model object
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        arch->set(splitFeature);
        arch->setSharedPtrObj(values);
    }
};
} // namespace interface1
using interface1::Model;

}
/** @} */
}
} // namespace daal
#endif
