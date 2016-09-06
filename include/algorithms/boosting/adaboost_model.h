/* file: adaboost_model.h */
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
//  Implementation of class defining Ada Boost model.
//--
*/

#ifndef __ADA_BOOST_MODEL_H__
#define __ADA_BOOST_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/boosting/boosting_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the AdaBoost classification algorithm
 */
namespace adaboost
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup adaboost
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__ADABOOST__PARAMETER"></a>
 * \brief AdaBoost algorithm parameters
 *
 * \snippet boosting/adaboost_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public boosting::Parameter
{
    /** Default constructor */
    Parameter();

    /**
     * Constructs the AdaBoost parameter structure
     * \param[in] wlTrain       Pointer to the training algorithm of the weak learner
     * \param[in] wlPredict     Pointer to the prediction algorithm of the weak learner
     * \param[in] acc           Accuracy of the AdaBoost training algorithm
     * \param[in] maxIter       Maximal number of iterations of the AdaBoost training algorithm
     */
    Parameter(services::SharedPtr<weak_learner::training::Batch>   wlTrain,
              services::SharedPtr<weak_learner::prediction::Batch> wlPredict,
              double acc = 0.0, size_t maxIter = 10);

    double accuracyThreshold;       /*!< Accuracy of the AdaBoost training algorithm */
    size_t maxIterations;           /*!< Maximal number of iterations of the AdaBoost training algorithm */

    void check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__MODEL"></a>
 * \brief %Model of the classifier trained by the adaboost::training::Batch algorithm.
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public boosting::Model
{
public:
    DAAL_DOWN_CAST_OPERATOR(Model,classifier::Model)

    /**
     * Constructs the AdaBoost model
     * \tparam modelFPType  Data type to store AdaBoost model data, double or float
     * \param[in] dummy     Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    DAAL_EXPORT Model(modelFPType dummy);

    /**
     * Empty constructor for deserialization
     */
    Model() : boosting::Model(), _alpha()
    {}

    virtual ~Model()
    {}

    /**
     *  Returns a pointer to the array of weights of weak learners constructed
     *  during training of the AdaBoost algorithm.
     *  The size of the array equals the number of weak learners
     *  \return Array of weights of weak learners.
     */
    data_management::NumericTablePtr getAlpha() const;

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_ADABOOST_MODEL_ID; }
    /**
     *  Serializes the AdaBoost model object
     *  \param[in]  archive  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Deserializes the AdaBoost model object
     *  \param[in]  archive  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    data_management::NumericTablePtr _alpha;     /* Boosting coefficients table */

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        boosting::Model::serialImpl<Archive, onDeserialize>(arch);
        arch->setSharedPtrObj(_alpha);
    }
}; // class Model
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Model;

} // namespace daal::algorithms::adaboost
}
} // namespace daal
#endif
