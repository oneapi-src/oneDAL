/* file: stump_model.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "data_management/data/matrix.h"
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
 * \brief %Model of the classifier trained by the stump::training::Batch algorithm.
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public weak_learner::Model
{
public:
    DAAL_DOWN_CAST_OPERATOR(Model, classifier::Model)
    DECLARE_SERIALIZABLE();

    /**
     * Constructs the decision stump model
     * \tparam modelFPType  Data type to store decision stump model data, double or float
     * \param[in] dummy     Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    DAAL_EXPORT Model(modelFPType dummy);

    /**
     * Empty constructor for deserialization
     */
    Model();

    /**
     *  Returns the split feature
     *  \return Index of the feature over which the split is made
     */
    size_t getSplitFeature();

    /**
     *  Sets the split feature
     *  \param[in] splitFeature   Index of the split feature
     */
    void setSplitFeature(size_t splitFeature);

    /**
     *  Returns a value of the feature that defines the split
     *  \return Value of the feature over which the split is made
     */
    template<typename modelFPType>
    DAAL_EXPORT modelFPType getSplitValue();

    /**
     *  Sets a value of the feature that defines the split
     *  \param[in] splitValue   Value of the split feature
     */
    template<typename modelFPType>
    DAAL_EXPORT void setSplitValue(modelFPType splitValue);

    /**
     *  Returns an average of the weighted responses for the "left" subset
     *  \return Average of the weighted responses for the "left" subset
     */
    template<typename modelFPType>
    DAAL_EXPORT modelFPType getLeftSubsetAverage();

    /**
     *  Sets an average of the weighted responses for the "left" subset
     *  \param[in] leftSubsetAverage   An average of the weighted responses for the "left" subset
     */
    template<typename modelFPType>
    DAAL_EXPORT void setLeftSubsetAverage(modelFPType leftSubsetAverage);

    /**
     *  Returns an average of the weighted responses for the "right" subset
     *  \return Average of the weighted responses for the "right" subset
     */
    template<typename modelFPType>
    DAAL_EXPORT modelFPType getRightSubsetAverage();

    /**
     *  Sets an average of the weighted responses for the "right" subset
     *  \param[in] rightSubsetAverage   An average of the weighted responses for the "right" subset
     */
    template<typename modelFPType>
    DAAL_EXPORT void setRightSubsetAverage(modelFPType rightSubsetAverage);

protected:
    size_t      _splitFeature;                                        /*!< Index of the feature over which the split is made */
    services::SharedPtr<data_management::Matrix<double> > _values;     /*!< Table that contains 3 values:\n
                                                                        Value of the feature that defines the split,\n
                                                                        Average of the weighted responses for the "left" subset,\n
                                                                        Average of the weighted responses for the "right" subset */

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        arch->set(_splitFeature);
        arch->setSharedPtrObj(_values);
    }
    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}
};
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;

}
/** @} */
}
} // namespace daal
#endif
