/* file: adaboost_training_types.h */
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
//  Implementation of Ada Boost training algorithm interface.
//--
*/

#ifndef __ADA_BOOST_TRAINING_TYPES_H__
#define __ADA_BOOST_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/boosting/adaboost_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup boosting Boosting
 * @ingroup classification
 * @defgroup adaboost Adaboost Classifier
 * \copydoc daal::algorithms::adaboost
 * @ingroup boosting
 */
namespace adaboost
{
/**
 * @defgroup adaboost_training Training
 * \copydoc daal::algorithms::adaboost::training
 * @ingroup adaboost
 * @{
 */
/**
 * \brief Contains classes for AdaBoost models training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ADABOOST__TRAINING__METHOD"></a>
 * Available methods for AdaBoost model training
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
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method
 *        of the AdaBoost training algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    Result() {}
    virtual ~Result() {}

    /**
     * Allocates memory to store final results of AdaBoost training
     * \param[in] input         %Input of the AdaBoost training algorithm
     * \param[in] parameter     Parameters of the algorithm
     * \param[in] method        AdaBoost computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns the model trained with the AdaBoost algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the AdaBoost algorithm
     */
    services::SharedPtr<daal::algorithms::adaboost::Model> get(classifier::training::ResultId id) const;

    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_ADABOOST_TRAINING_RESULT_ID; }

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

} // namespace daal::algorithms::adaboost::training
/** @} */
}
}
} // namespace daal
#endif // __ADA_BOOST_TRAINING_TYPES_H__
