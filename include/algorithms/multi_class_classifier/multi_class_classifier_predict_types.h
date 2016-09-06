/* file: multi_class_classifier_predict_types.h */
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

#ifndef __MULTI_CLASS_CLASSIFIER_PREDICT_TYPES_H__
#define __MULTI_CLASS_CLASSIFIER_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
/**
 * @defgroup multi_class_classifier_prediction Prediction
 * \copydoc daal::algorithms::multi_class_classifier::prediction
 * @ingroup multi_class_classifier
 * @{
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__METHOD"></a>
 * Computation methods for the multi-class classifier prediction algorithm
 */
enum Method
{
    defaultDense = 0,           /*!< Prediction method for the multi-class classifier proposed by Ting-Fan Wu et al. */
    multiClassClassifierWu = 0  /*!< Prediction method for the multi-class classifier proposed by Ting-Fan Wu et al. */
};

namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the Multi-class classifier algorithm
 */
class Input : public classifier::prediction::Input
{
public:
    Input() {}
    virtual ~Input() {}

    /**
     * Returns the input Numeric Table object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(classifier::prediction::NumericTableInputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the input Model object in the prediction stage of the Multi-class classifier algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<multi_class_classifier::Model> get(classifier::prediction::ModelInputId id) const
    {
        return services::staticPointerCast<multi_class_classifier::Model, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets the input Model object in the prediction stage of the Multi-class classifier algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::ModelInputId id, const services::SharedPtr<multi_class_classifier::Model> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        classifier::prediction::Input::check(parameter, method);
        if(this->_errors->size() != 0) { return; }

        services::SharedPtr<multi_class_classifier::Model> m = get(classifier::prediction::model);
        if(m->getNumberOfTwoClassClassifierModels() == 0)
        {
            this->_errors->add(services::ErrorModelNotFullInitialized);
            return;
        }
    }

};

} // namespace interface1
using interface1::Input;

} // namespace prediction
/** @} */
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
#endif
