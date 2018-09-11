/* file: boosting_model.h */
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
//  Implementation of the base class defining Boosting algorithm model.
//--
*/

#ifndef __BOOSTING_MODEL_H__
#define __BOOSTING_MODEL_H__

#include "algorithms/weak_learner/weak_learner_model.h"
#include "algorithms/weak_learner/weak_learner_training_batch.h"
#include "algorithms/weak_learner/weak_learner_predict.h"
#include "algorithms/stump/stump_training_batch.h"
#include "algorithms/stump/stump_predict.h"
#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of boosting classification algorithms
 */
namespace boosting
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup boosting
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__BOOSTING__PARAMETER"></a>
* \brief %Base class for parameters of the %boosting algorithm
 *
 * \snippet boosting/boosting_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    /** Default constructor. Sets the decision stump as the default weak learner */
    Parameter();

    /**
     * Constructs %boosting algorithm parameters from weak learner training and prediction algorithms
     * \param[in] wlTrainForParameter       Pointer to the training algorithm of the weak learner
     * \param[in] wlPredictForParameter     Pointer to the prediction algorithm of the weak learner
     */
    Parameter(const services::SharedPtr<weak_learner::training::Batch>&   wlTrainForParameter,
              const services::SharedPtr<weak_learner::prediction::Batch>& wlPredictForParameter);

    /** Copy constructor */
    Parameter(const Parameter& other) :weakLearnerTraining(other.weakLearnerTraining),
        weakLearnerPrediction(other.weakLearnerPrediction){}

    /** The algorithm for weak learner model training */
    services::SharedPtr<weak_learner::training::Batch>   weakLearnerTraining;

    /** The algorithm for prediction based on a weak learner model */
    services::SharedPtr<weak_learner::prediction::Batch> weakLearnerPrediction;

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */
/** @} */

/**
 * @ingroup boosting
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BOOSTING__MODEL"></a>
* \brief %Base class for %boosting algorithm models.
 *        Contains a collection of weak learner models constructed during training of the %boosting algorithm
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    /**
     * Constructs the model trained with the boosting algorithm
     * \param[in] nFeatures Number of features in the dataset
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model(size_t nFeatures = 0) : _models(new data_management::DataCollection()), _nFeatures(nFeatures) {}

    virtual ~Model() {}

    /**
     *  Returns the number of weak learners constructed during training of the %boosting algorithm
     *  \return The number of weak learners
     */
    size_t getNumberOfWeakLearners() const;

    /**
     *  Returns weak learner model constructed during training of the %boosting algorithm
     *  \param[in] idx  Index of the model in the collection
     *  \return Weak Learner model corresponding to the index idx
     */
    weak_learner::ModelPtr getWeakLearnerModel(size_t idx) const;

    /**
     *  Add weak learner model into the %boosting model
     *  \param[in] model Weak learner model to add into collection
     */
    void addWeakLearnerModel(weak_learner::ModelPtr model);

    void clearWeakLearnerModels();

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return _nFeatures; }

protected:
    size_t _nFeatures;
    data_management::DataCollectionPtr _models;

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        arch->set(_nFeatures);
        arch->setSharedPtrObj(_models);

        return services::Status();
    }

    Model(size_t nFeatures, services::Status &st);
};
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace daal::algorithms::boosting
}
}
#endif // __BOOSTING_MODEL_H__
