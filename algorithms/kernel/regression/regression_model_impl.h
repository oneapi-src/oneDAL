/* file: regression_model_impl.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of the class defining the regression model
//--
*/

#ifndef __REGRESSION_MODEL_IMPL_H__
#define __REGRESSION_MODEL_IMPL_H__

#include "algorithms/regression/regression_model.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace internal
{

class ModelInternal
{
public:
    ModelInternal(size_t nFeatures = 0) : _nFeatures(nFeatures)
    {}

    void setNumberOfFeatures(size_t nFeatures) { _nFeatures = nFeatures; }
    virtual size_t getNumberOfFeatures() const { return _nFeatures; }
protected:
    size_t _nFeatures;

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(_nFeatures);

        return services::Status();
    }
};

class ModelImpl : public regression::Model,
                  public ModelInternal
{
public:
    typedef ModelInternal   ImplType;

    ModelImpl(size_t nFeatures = 0) : ImplType(nFeatures)
    {}

    void setNumberOfFeatures(size_t nFeatures) { ImplType::setNumberOfFeatures(nFeatures); }
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return ImplType::getNumberOfFeatures(); }

protected:

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        auto s = regression::Model::serialImpl<Archive, onDeserialize>(arch);
        return s.add(ModelInternal::serialImpl<Archive, onDeserialize>(arch));
    }
};

}
}
}
}

#endif
