/* file: stump_model.h */
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
//  Implementation of the class defining the decision stump model.
//--
*/

#ifndef __STUMP_MODEL_H__
#define __STUMP_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/matrix.h"
#include "algorithms/weak_learner/weak_learner_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup stump Stump
 * \copydoc daal::algorithms::stump
 * @ingroup weak_learner
 * @{
 */
namespace stump
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__MODEL"></a>
 * \brief %Model of the classifier trained by the stump::training::Batch algorithm.
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public weak_learner::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model);

    /**
     * Constructs the decision stump model
     * \tparam modelFPType  Data type to store decision stump model data, double or float
     * \param[in] nFeatures Number of features in the dataset
     * \param[in] dummy     Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template<typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, modelFPType dummy);

    /**
     * Constructs the decision stump model
     * \tparam modelFPType  Data type to store decision stump model data, double or float
     * \param[in]  nFeatures Number of features in the dataset
     * \param[out] stat      Status of the model construction
     * \return Decision stump model
     */
    template<typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<Model> create(size_t nFeatures, services::Status *stat = NULL);

    /**
     * Empty constructor for deserialization
     */
    Model();

    /**
     *  Returns the split feature
     *  \return Index of the feature over which the split is made
     */
    size_t getSplitFeature();

    /**
     *  Sets the split feature
     *  \param[in] splitFeature   Index of the split feature
     */
    void setSplitFeature(size_t splitFeature);

    /**
     *  Returns a value of the feature that defines the split
     *  \return Value of the feature over which the split is made
     */
    template<typename modelFPType>
    DAAL_EXPORT modelFPType getSplitValue();

    /**
     *  Sets a value of the feature that defines the split
     *  \param[in] splitValue   Value of the split feature
     */
    template<typename modelFPType>
    DAAL_EXPORT void setSplitValue(modelFPType splitValue);

    /**
     *  Returns an average of the weighted responses for the "left" subset
     *  \return Average of the weighted responses for the "left" subset
     */
    template<typename modelFPType>
    DAAL_EXPORT modelFPType getLeftSubsetAverage();

    /**
     *  Sets an average of the weighted responses for the "left" subset
     *  \param[in] leftSubsetAverage   An average of the weighted responses for the "left" subset
     */
    template<typename modelFPType>
    DAAL_EXPORT void setLeftSubsetAverage(modelFPType leftSubsetAverage);

    /**
     *  Returns an average of the weighted responses for the "right" subset
     *  \return Average of the weighted responses for the "right" subset
     */
    template<typename modelFPType>
    DAAL_EXPORT modelFPType getRightSubsetAverage();

    /**
     *  Sets an average of the weighted responses for the "right" subset
     *  \param[in] rightSubsetAverage   An average of the weighted responses for the "right" subset
     */
    template<typename modelFPType>
    DAAL_EXPORT void setRightSubsetAverage(modelFPType rightSubsetAverage);

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return _nFeatures; }

protected:
    size_t      _nFeatures;                                            /*!< Number of features in the dataset was used
                                                                            on the training stage */
    size_t      _splitFeature;                                         /*!< Index of the feature over which the split is made */
    services::SharedPtr<data_management::Matrix<double> > _values;     /*!< Table that contains 3 values:\n
                                                                        Value of the feature that defines the split,\n
                                                                        Average of the weighted responses for the "left" subset,\n
                                                                        Average of the weighted responses for the "right" subset */

    template<typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, modelFPType dummy, services::Status &st);

    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        services::Status st = classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st)
            return st;
        arch->set(_nFeatures);
        arch->set(_splitFeature);
        arch->setSharedPtrObj(_values);

        return st;
    }
};
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;

}
/** @} */
}
} // namespace daal
#endif
