/* file: classifier_model.h */
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
//  Implementation of the class defining the model of the classification  algorithm
//--
*/

#ifndef __CLASSIFIER_MODEL_H__
#define __CLASSIFIER_MODEL_H__

#include "algorithms/algorithm.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup classifier
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__CLASSIFIER__PARAMETER"></a>
 * \brief Base class for the parameters of the classification algorithm
 *
 * \snippet classifier/classifier_model.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nClasses = 2) : nClasses(nClasses) {}

    size_t nClasses;        /*!< Number of classes */

    void check() const DAAL_C11_OVERRIDE
    {
        if(nClasses == 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nClassesStr()));
            return;
        }
    }
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__MODEL"></a>
 * \brief Base class for the model of the classification algorithm
 */
class Model : public daal::algorithms::Model
{
public:
    /** Default constructor */
    Model() : _nFeatures(0), daal::algorithms::Model() {}
    virtual ~Model() {}

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNFeatures() const { return _nFeatures; }

    /**
     *  Sets the number of features in the dataset was used on the training stage
     *  \param[in]  nFeatures  Number of features in the dataset was used on the training stage
     */
    void setNFeatures(size_t nFeatures) { _nFeatures = nFeatures; }

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
    size_t _nFeatures;

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        algorithms::Model::serialImpl<Archive, onDeserialize>(arch);
        arch->set(_nFeatures);
    }
};
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Model;

}
}
}
#endif
