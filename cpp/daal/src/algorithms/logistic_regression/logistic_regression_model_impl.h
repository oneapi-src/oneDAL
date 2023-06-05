/* file: logistic_regression_model_impl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the class defining the logistic regression model
//--
*/

#ifndef __LOGISTIC_REGRESSION_MODEL_IMPL__
#define __LOGISTIC_REGRESSION_MODEL_IMPL__

#include "algorithms/logistic_regression/logistic_regression_model.h"
#include "src/algorithms/classifier/classifier_model_impl.h"

#include <iostream>

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace internal
{
class ModelImpl : public logistic_regression::Model, public algorithms::classifier::internal::ModelInternal
{
public:
    typedef algorithms::classifier::internal::ModelInternal ClassificationImplType;

    ModelImpl(size_t nFeatures = 0, bool interceptFlag = true);
    template <typename modelFPType>
    ModelImpl(size_t nFeatures, bool interceptFlag, size_t nClasses, modelFPType dummy, services::Status * st);
    ~ModelImpl() {}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return ClassificationImplType::getNumberOfFeatures(); }

    //Implementation of classification::Model
    virtual size_t getNumberOfBetas() const DAAL_C11_OVERRIDE;
    virtual bool getInterceptFlag() const DAAL_C11_OVERRIDE;
    virtual data_management::NumericTablePtr getBeta() DAAL_C11_OVERRIDE;
    virtual const data_management::NumericTablePtr getBeta() const DAAL_C11_OVERRIDE;

    virtual services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

    services::Status reset(bool interceptFlag);
    static logistic_regression::ModelPtr create(size_t nFeatures, bool interceptFlag, services::Status * stat = nullptr);

protected:
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {   
        std::cout << "ModelImpl::serialImpl()" << std::endl;
        auto st = classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st) return st;
        std::cout << "classifier::Model::serialImpl<Archive, onDeserialize>(arch);" << std::endl;
        arch->set(ClassificationImplType::_nFeatures);
        std::cout << "arch->set(ClassificationImplType::_nFeatures);" << std::endl;
        arch->set(_interceptFlag);
        std::cout << "arch->set(_interceptFlag);" << std::endl;
        arch->setSharedPtrObj(_beta);
        std::cout << "arch->setSharedPtrObj(_beta);" << std::endl;
        return services::Status();
    }

protected:
    bool _interceptFlag;                    // True if the model contains the intercept term false otherwise
    data_management::NumericTablePtr _beta; // Model coefficients
};

} // namespace internal
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

#endif
