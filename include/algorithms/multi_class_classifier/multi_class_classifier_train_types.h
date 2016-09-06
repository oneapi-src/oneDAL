/* file: multi_class_classifier_train_types.h */
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
//  Multiclass classifier data types
//--
*/

#ifndef __MULTI_CLASS_CLASSIFIER_TRAIN_TYPES_H__
#define __MULTI_CLASS_CLASSIFIER_TRAIN_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
/**
 * @defgroup multi_class_classifier_training Training
 * \copydoc daal::algorithms::multi_class_classifier::training
 * @ingroup multi_class_classifier
 * @{
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTI_CLASS_CLASSIFIER__TRAINING__METHOD"></a>
 * Available computation methods for the multi-class classifier algorithm
 */
enum Method
{
    oneAgainstOne    = 0   /*!< One-against-one method */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method for the
 *        multi-class classifier algorithm in the batch processing mode;
 *        or finalizeCompute() method of the algorithm in the online or distributed processing mode
 */
class Result : public classifier::training::Result
{
public:
    Result() {}

    virtual ~Result() {}

    /**
     * Returns the model trained with the Multi class classifier algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the Multi class classifier algorithm
     */
    services::SharedPtr<daal::algorithms::multi_class_classifier::Model> get(classifier::training::ResultId id) const
    {
        return services::staticPointerCast<daal::algorithms::multi_class_classifier::Model, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Registers user-allocated memory to store the results of the multi-class classifier training decomposition
     * \param[in] input       Pointer to the structure with input objects
     * \param[in] parameter   Pointer to the structure with algorithm parameters
     * \param[in] method      Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const ParameterBase *algParameter = static_cast<const ParameterBase *>(parameter);
        set(classifier::training::model, services::SharedPtr<classifier::Model>(new Model(algParameter)));
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_MULTICLASS_CLASSIFIER_RESULT_ID; }

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

    /**
     * Checks the correctness of the Result object
     * \param[in] input     Pointer to the structure of the input objects
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        checkImpl(input, parameter);
        if(this->_errors->size() != 0) { return; }
        const Parameter *par = static_cast<const Parameter *>(parameter);
        services::SharedPtr<daal::algorithms::multi_class_classifier::Model> m = get(classifier::training::model);
        if(m->getNumberOfTwoClassClassifierModels() == 0)
        {
            this->_errors->add(services::ErrorModelNotFullInitialized);
            return;
        }
        if(m->getNumberOfTwoClassClassifierModels() != par->nClasses * (par->nClasses - 1) / 2)
        {
            this->_errors->add(services::ErrorModelNotFullInitialized);
            return;
        }
    }
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

} // namespace training
/** @} */
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
#endif
