/* file: multi_class_classifier_model.h */
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
//  Multiclass tcc parameter structure
//--
*/

#ifndef __MULTI_CLASS_CLASSIFIER_MODEL_H__
#define __MULTI_CLASS_CLASSIFIER_MODEL_H__

#include "services/daal_defines.h"
#include "algorithms/model.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/classifier/classifier_predict.h"
#include "algorithms/classifier/classifier_training_batch.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup multi_class_classifier Multi-class Classifier
 * \copydoc daal::algorithms::multi_class_classifier
 * @ingroup classification
 * @{
 */
/**
 * \brief Contains classes for computing the results of the multi-class classifier algorithm
 */
namespace multi_class_classifier
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup multi_class_classifier
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PARAMETERBASE"></a>
 * \brief Parameters of the multi-class classifier algorithm
 *
 * \snippet multi_class_classifier/multi_class_classifier_model.h ParameterBase source code
 */
/* [ParameterBase source code] */
struct DAAL_EXPORT ParameterBase : public daal::algorithms::classifier::Parameter
{
    ParameterBase(size_t nClasses): daal::algorithms::classifier::Parameter(nClasses), training(), prediction() {}
    services::SharedPtr<classifier::training::Batch> training;          /*!< Two-class classifier training stage */
    services::SharedPtr<classifier::prediction::Batch> prediction;      /*!< Two-class classifier prediction stage */

    void check() const DAAL_C11_OVERRIDE
    {
        classifier::Parameter::check();
        if(!training.get())
        {
            this->_errors->add(services::Error::create(services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, trainingStr()));
            return;
        }
        if(!prediction.get())
        {
            this->_errors->add(services::Error::create(services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, predictionStr()));
            return;
        }
    }
};
/* [ParameterBase source code] */

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PARAMETER"></a>
 * \brief Optional multi-class classifier algorithm  parameters that are used with the MultiClassClassifierWu prediction method
 *
 * \snippet multi_class_classifier/multi_class_classifier_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public ParameterBase
{
    Parameter(size_t nClasses, size_t maxIterations = 100, double accuracyThreshold = 1.0e-12) :
        ParameterBase(nClasses), maxIterations(maxIterations), accuracyThreshold(accuracyThreshold) {}

    size_t maxIterations;     /*!< Maximum number of iterations */
    double accuracyThreshold; /*!< Convergence threshold */

    void check() const DAAL_C11_OVERRIDE
    {
        ParameterBase::check();
        if(accuracyThreshold <= 0 || accuracyThreshold >= 1)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, accuracyThresholdStr()));
            return;
        }
        if(maxIterations == 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr()));
            return;
        }
    }
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__MODEL"></a>
 * \brief Model of the classifier trained by the multi_class_classifier::training::Batch algorithm.
 */
class Model : public daal::algorithms::classifier::Model
{
public:
    DAAL_DOWN_CAST_OPERATOR(Model,classifier::Model)

    /**
     *  Constructs multi-class classifier model
     *  \param[in] par Parameters of the multi-class classifier algorithm
     */
    Model(const ParameterBase *par) : _modelsArray(NULL), _models(new data_management::DataCollection(par->nClasses * (par->nClasses - 1) / 2))
    {}

    /**
     * Empty constructor for deserialization
     */
    Model() : _modelsArray(NULL), _models(new data_management::DataCollection()) {}

    ~Model()
    {
        if(_modelsArray) { delete [] _modelsArray; }
    }

    /**
     *  Returns a collection of two-class classifier models in a multi-class classifier model
     *  \return  Collection of two-class classifier models
     */
    data_management::DataCollectionPtr getMultiClassClassifierModel()
    {
        return _models;
    }

    /**
     *  Returns a pointer to the array of two-class classifier models in a multi-class classifier model
     *  \return Pointer to the array of two-class classifier models
     */
    services::SharedPtr<classifier::Model> *getTwoClassClassifierModels()
    {
        if(!_modelsArray)
        {
            _modelsArray = new services::SharedPtr<classifier::Model>[_models->size()];
            for(size_t i = 0; i < _models->size(); i++)
            {
                _modelsArray[i] = services::staticPointerCast<classifier::Model, data_management::SerializationIface>((*_models)[i]);
            }
        }
        return _modelsArray;
    }

    /**
     *  Set two-class classifier model into a multi-class classifier model
     *  \param[in] idx    Index of two-class classifier model in a collection
     *  \param[in] model  Two-class classifier model to add into collection
     */
    void setTwoClassClassifierModel(size_t idx, services::SharedPtr<classifier::Model> model)
    {
        (*_models)[idx] = model;
    }

    /**
     *  Returns a two-class classifier model in a multi-class classifier model
     *  \param[in]  idx   Index of the two-class classifier model in a multi-class classifier model
     *  \return             Two-class classifier model
     */
    services::SharedPtr<classifier::Model> getTwoClassClassifierModel(size_t idx) const
    {
        if(idx < _models->size())
        {
            return services::staticPointerCast<classifier::Model, data_management::SerializationIface>((*_models)[idx]);
        }
        return services::SharedPtr<classifier::Model>();
    }

    /**
     *  Returns a number of two-class classifiers associated with the model
     *  \return        Number of two-class classifiers associated with the model
     */
    size_t getNumberOfTwoClassClassifierModels()
    {
        return _models->size();
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_MULTI_CLASS_CLASSIFIER_MODEL_ID; }
    /**
     *  Implements serialization of the multi-class classifier model object
     *  \param[in]  archive  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Implements deserialization of the multi-class classifier model object
     *  \param[in]  archive  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    data_management::DataCollectionPtr _models;              /* Collection of two-class classifiers associated with the model */
    services::SharedPtr<classifier::Model> *_modelsArray;

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        arch->setSharedPtrObj(_models);
    }
};
/** @} */
} // namespace interface1
using interface1::ParameterBase;
using interface1::Parameter;
using interface1::Model;

} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
#endif
