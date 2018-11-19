/* file: naivebayes_model_fpt.cpp */
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
//  Implementation of class defining naive bayes algorithm model
//--
*/

#include "algorithms/naive_bayes/multinomial_naive_bayes_model.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{

/**
 * Constructs multinomial naive Bayes model
 * \param[in] nFeatures  The number of features
 * \param[in] parameter  The multinomial naive Bayes parameter
 * \param[in] dummy      Dummy variable for the templated constructor
 * \DAAL_DEPRECATED_USE{ Model::create }
 */
template<typename modelFPType>
DAAL_EXPORT Model::Model(size_t nFeatures, const Parameter &parameter, modelFPType dummy)
{
    using namespace data_management;

    const Parameter *par = &parameter;
    if(par->nClasses < 2 || nFeatures == 0)
    {
        return;
    }

    _logP     = NumericTablePtr(new HomogenNumericTable<modelFPType>(1,         par->nClasses, NumericTable::doAllocate));
    _logTheta = NumericTablePtr(new HomogenNumericTable<modelFPType>(nFeatures, par->nClasses, NumericTable::doAllocate));
    _auxTable = NumericTablePtr(new HomogenNumericTable<modelFPType>(nFeatures, par->nClasses, NumericTable::doAllocate));
}

template<typename modelFPType>
DAAL_EXPORT Model::Model(size_t nFeatures, const Parameter &parameter, modelFPType dummy, services::Status &st)
{
    using namespace data_management;

    if (parameter.nClasses < 2)
    {
        st.add(services::ErrorIncorrectNumberOfClasses);
        return;
    }
    if (nFeatures == 0)
    {
        st.add(services::ErrorIncorrectNumberOfFeatures);
        return;
    }

    _logP     = HomogenNumericTable<modelFPType>::create(1,         parameter.nClasses, NumericTable::doAllocate, &st);
    if (!st)
        return;
    _logTheta = HomogenNumericTable<modelFPType>::create(nFeatures, parameter.nClasses, NumericTable::doAllocate, &st);
    if (!st)
        return;
    _auxTable = HomogenNumericTable<modelFPType>::create(nFeatures, parameter.nClasses, NumericTable::doAllocate, &st);
    if (!st)
        return;
}

/**
 * Constructs multinomial naive Bayes model
 * \param[in] nFeatures  The number of features
 * \param[in] parameter  The multinomial naive Bayes parameter
 * \param[out] stat      Status of the model construction
 */
template<typename modelFPType>
DAAL_EXPORT ModelPtr Model::create(size_t nFeatures, const Parameter &parameter, services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures, parameter, (modelFPType)0.0);
}

/**
 * Constructs multinomial naive Bayes partial model
 * \param[in] nFeatures  The number of features
 * \param[in] parameter  Multinomial naive Bayes parameter
 * \param[in] dummy      Dummy variable for the templated constructor
 * \DAAL_DEPRECATED_USE{ PartialModel::create }
 */
template<typename modelFPType>
DAAL_EXPORT PartialModel::PartialModel(size_t nFeatures, const Parameter &parameter, modelFPType dummy) : _nObservations(0)
{
    using namespace data_management;
    const Parameter *par = &parameter;
    if(par->nClasses < 2 || nFeatures == 0)
    {
        return;
    }

    _classSize     = NumericTablePtr(new HomogenNumericTable<int>(1,         par->nClasses, NumericTable::doAllocate));
    _classGroupSum = NumericTablePtr(new HomogenNumericTable<int>(nFeatures, par->nClasses, NumericTable::doAllocate));
}

template<typename modelFPType>
DAAL_EXPORT PartialModel::PartialModel(size_t nFeatures, const Parameter &parameter,
                                       modelFPType dummy, services::Status &st) : _nObservations(0)
{
    using namespace data_management;

    if (parameter.nClasses < 2)
    {
        st.add(services::ErrorIncorrectNumberOfClasses);
        return;
    }
    if (nFeatures == 0)
    {
        st.add(services::ErrorIncorrectNumberOfFeatures);
        return;
    }

    _classSize     = HomogenNumericTable<int>::create(1,         parameter.nClasses, NumericTable::doAllocate, &st);
    if (!st)
        return;

    _classGroupSum = HomogenNumericTable<int>::create(nFeatures, parameter.nClasses, NumericTable::doAllocate, &st);
    if (!st)
        return;
}

/**
 * Constructs multinomial naive Bayes partial model
 * \param[in] nFeatures  The number of features
 * \param[in] parameter  The multinomial naive Bayes parameter
 * \param[out] stat      Status of the model construction
 * \return Multinomial naive Bayes partial model
 */
template<typename modelFPType>
DAAL_EXPORT PartialModelPtr PartialModel::create(size_t nFeatures, const Parameter &parameter, services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(PartialModel, nFeatures, parameter, (modelFPType)0.0);
}

template DAAL_EXPORT Model::Model(size_t, const Parameter&, DAAL_FPTYPE);
template DAAL_EXPORT Model::Model(size_t, const Parameter&, DAAL_FPTYPE, services::Status&);

template DAAL_EXPORT ModelPtr Model::create<DAAL_FPTYPE>(size_t, const Parameter&, services::Status*);

template DAAL_EXPORT PartialModel::PartialModel(size_t, const Parameter&, DAAL_FPTYPE);
template DAAL_EXPORT PartialModel::PartialModel(size_t, const Parameter&, DAAL_FPTYPE, services::Status&);

template DAAL_EXPORT PartialModelPtr PartialModel::create<DAAL_FPTYPE>(size_t, const Parameter&, services::Status*);

}// namespace multinomial_naive_bayes
}// namespace algorithms
}// namespace daal
