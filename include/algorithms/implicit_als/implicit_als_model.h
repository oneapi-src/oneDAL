/* file: implicit_als_model.h */
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
//  Declaration of the implicit ALS model class
//--
*/

#ifndef __IMPLICIT_ALS_MODEL_H__
#define __IMPLICIT_ALS_MODEL_H__

#include "algorithms/model.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup recommendation_systems Recommendation Systems
 * @ingroup training_and_prediction
 * @defgroup implicit_als Implicit Alternating Least Squares
 * \copydoc daal::algorithms::implicit_als
 * @ingroup recommendation_systems
 * @{
 */
namespace implicit_als
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__IMPLICIT_ALS__PARAMETER"></a>
 * \brief Parameters for the compute() method of the implicit ALS algorithm
 *
 * \snippet implicit_als/implicit_als_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nFactors = 10, size_t maxIterations = 5, double alpha = 40.0, double lambda = 0.01,
              double preferenceThreshold = 0.0, size_t seed = 777777) :
        nFactors(nFactors), maxIterations(maxIterations), alpha(alpha), lambda(lambda),
        preferenceThreshold(preferenceThreshold)
    {}

    size_t nFactors;            /*!< Number of factors */
    size_t maxIterations;       /*!< Maximum number of iterations of the implicit ALS training algorithm */
    double alpha;               /*!< Confidence parameter of the implicit ALS training algorithm */
    double lambda;              /*!< Regularization parameter */
    double preferenceThreshold; /*!< Threshold used to define preference values */

    void check() const DAAL_C11_OVERRIDE
    {
        if(nFactors == 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nFactorsStr()));
            return;
        }
        if(maxIterations == 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr()));
            return;
        }
        if(alpha < 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, alphaStr()));
            return;
        }
        if(lambda < 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, lambdaStr()));
            return;
        }
        if(preferenceThreshold < 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, preferenceThresholdStr()));
            return;
        }
    }
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__MODEL"></a>
 * \brief Model trained by the implicit ALS algorithm in the batch processing mode
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - Parameter class
 */
class Model : public daal::algorithms::Model
{
public:
    /**
     * Constructs the implicit ALS model
     * \param[in] nUsers    Number of users in the input data set
     * \param[in] nItems    Number of items in the input data set
     * \param[in] parameter Implicit ALS parameters
     * \param[in] dummy     Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    Model(size_t nUsers, size_t nItems, const Parameter &parameter, modelFPType dummy)
    {
        size_t nFactors = parameter.nFactors;

        _usersFactors = data_management::NumericTablePtr(
                    new data_management::HomogenNumericTable<modelFPType>(
                            nFactors, nUsers, data_management::NumericTableIface::doAllocate, 0));
        _itemsFactors = data_management::NumericTablePtr(
                    new data_management::HomogenNumericTable<modelFPType>(
                            nFactors, nItems, data_management::NumericTableIface::doAllocate, 0));
    }

    /**
     * Empty constructor for deserialization
     */
    Model()
    {}

    virtual ~Model()
    {}

    /**
     * Returns a pointer to the numeric table of users factors constructed during the training
     * of the implicit ALS model
     * \return Numeric table of users factors
     */
    data_management::NumericTablePtr getUsersFactors() const { return _usersFactors; }

    /**
     * Returns a pointer to the numeric table of items factors constructed during the training
     * of the implicit ALS model
     * \return Numeric table of items factors
     */
    data_management::NumericTablePtr getItemsFactors() const { return _itemsFactors; }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_MODEL_ID; }

    /**
     *  Serializes a model object
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Deserializes a model object
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    data_management::NumericTablePtr _usersFactors;    /* Table of resulting users factors */
    data_management::NumericTablePtr _itemsFactors;    /* Table of resulting items factors */

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_usersFactors);
        arch->setSharedPtrObj(_itemsFactors);
    }
};
typedef services::SharedPtr<Model> ModelPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PARTIALMODEL"></a>
 * \brief Partial model trained by the implicit ALS training algorithm in the distributed processing mode
 *
 * \par References
 *      - \ref training::interface1::Distributed "implicit_als::training::Distributed"
 *      - Parameter class
 */
class PartialModel : public daal::algorithms::Model
{
public:

    DAAL_CAST_OPERATOR(PartialModel);
    /**
     * Constructs a partial implicit ALS model of a specified size
     * \param[in] parameter Implicit ALS parameters
     * \param[in] size      Model size
     * \param[in] dummy     Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    PartialModel(const Parameter &parameter, size_t size, modelFPType dummy)
    {
        size_t nFactors = parameter.nFactors;

        _factors = data_management::NumericTablePtr(
                                  new data_management::HomogenNumericTable<modelFPType>(
                                      nFactors, size, data_management::NumericTableIface::doAllocate));
        data_management::HomogenNumericTable<int> *_indicesTable = new data_management::HomogenNumericTable<int>(
                                      1, size, data_management::NumericTableIface::doAllocate);
        _indices = data_management::NumericTablePtr(_indicesTable);
        int *indicesData = _indicesTable->getArray();
        int iSize = (int)size;
        for (int i = 0; i < iSize; i++)
        {
            indicesData[i] = i;
        }
    }

    /**
     * Constructs a partial implicit ALS model from the indices of factors
     * \param[in] parameter Implicit ALS parameters
     * \param[in] offset    Index of the first factor in the partial model
     * \param[in] indices   Pointer to the numeric table with the indices of factors
     * \param[in] dummy     Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    PartialModel(const Parameter &parameter, size_t offset,
                 data_management::NumericTablePtr indices, modelFPType dummy)
    {
        size_t nFactors = parameter.nFactors;

        data_management::BlockDescriptor<int> block;
        size_t size = indices->getNumberOfRows();
        indices->getBlockOfRows(0, size, data_management::readOnly, block);
        int *srcIndicesData = block.getBlockPtr();

        _factors = data_management::NumericTablePtr(
                                  new data_management::HomogenNumericTable<modelFPType>(
                                      nFactors, size, data_management::NumericTableIface::doAllocate));
        data_management::HomogenNumericTable<int> *_indicesTable = new data_management::HomogenNumericTable<int>(
                                      1, size, data_management::NumericTableIface::doAllocate);
        _indices = data_management::NumericTablePtr(_indicesTable);
        int *dstIndicesData = _indicesTable->getArray();
        int iOffset = (int)offset;
        for (size_t i = 0; i < size; i++)
        {
            dstIndicesData[i] = srcIndicesData[i] + iOffset;
        }
        indices->releaseBlockOfRows(block);
    }

    /**
     * Constructs a partial implicit ALS model from the indices and factors stored in the numeric tables

     * \param[in] factors   Pointer to the numeric table with factors stored in row-major order
     * \param[in] indices   Pointer to the numeric table with the indices of factors
     */
    PartialModel(data_management::NumericTablePtr factors,
                 data_management::NumericTablePtr indices) :
        _factors(factors), _indices(indices)
    {}

    /**
     * Empty constructor for deserialization
     */
    PartialModel()
    {}

    virtual ~PartialModel()
    {}

    /**
     * Returns pointer to the numeric table with factors stored in row-major order
     * \return Pointer to the numeric table with factors stored in row-major order
     */
    data_management::NumericTablePtr getFactors() const { return _factors; }

    /**
     * Returns the pointer to the numeric table with the indices of factors
     * \return Pointer to the numeric table with the indices of factors
     */
    data_management::NumericTablePtr getIndices() const { return _indices; }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID; }

    /**
     *  Serializes a model object
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Deserializes a model object
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    data_management::NumericTablePtr _factors;      /* Factors in row-major format */
    data_management::NumericTablePtr _indices;      /* Indices of the factors */

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_factors);
        arch->setSharedPtrObj(_indices);
    }
};

typedef services::SharedPtr<PartialModel> PartialModelPtr;
} // namespace interface1
using interface1::Parameter;
using interface1::ModelPtr;
using interface1::Model;
using interface1::PartialModelPtr;
using interface1::PartialModel;

}
/** @} */
}
}

#endif
