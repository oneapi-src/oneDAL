/* file: multiclassclassifier_model.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of multi class classifier model.
//--
*/

#include "multi_class_classifier_model.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_MULTI_CLASS_CLASSIFIER_MODEL_ID);

Model::Model(size_t nFeatures, const ParameterBase *par) :
    _modelsArray(NULL),
    _models(new data_management::DataCollection(par->nClasses * (par->nClasses - 1) / 2)),
    _nFeatures(nFeatures)
{
}

Model::Model(size_t nFeatures, const ParameterBase *par, services::Status &st) :
    _modelsArray(NULL),
    _nFeatures(nFeatures)
{
    _models.reset(new data_management::DataCollection(par->nClasses * (par->nClasses - 1) / 2));
    if (!_models)
        st.add(services::ErrorMemoryAllocationFailed);
}

Model::Model() : _modelsArray(NULL), _models(new data_management::DataCollection()), _nFeatures(0)
{
}

ModelPtr Model::create(size_t nFeatures, const ParameterBase *par, services::Status* stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures, par);
}

Model::~Model()
{
    delete[] _modelsArray;
}

classifier::ModelPtr *Model::getTwoClassClassifierModels()
{
    if(!_modelsArray)
    {
        _modelsArray = new classifier::ModelPtr[_models->size()];
        for(size_t i = 0; i < _models->size(); i++)
        {
            _modelsArray[i] = services::staticPointerCast<classifier::Model, data_management::SerializationIface>((*_models)[i]);
        }
    }
    return _modelsArray;
}

void Model::setTwoClassClassifierModel(size_t idx, const classifier::ModelPtr& model)
{
    (*_models)[idx] = model;
}

classifier::ModelPtr Model::getTwoClassClassifierModel(size_t idx) const
{
    if(idx < _models->size())
    {
        return services::staticPointerCast<classifier::Model, data_management::SerializationIface>((*_models)[idx]);
    }
    return classifier::ModelPtr();
}

services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, ParameterBase::check());
    DAAL_CHECK_EX((accuracyThreshold > 0) && (accuracyThreshold < 1), services::ErrorIncorrectParameter, services::ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations, services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr());
    return s;
}
}
}
}
}
