/* file: daal_factory_impl.cpp */
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
//  Implementation of dictionary utils.
//--
*/

#include "data_archive.h"
#include "homogen_numeric_table.h"
#include "aos_numeric_table.h"
#include "soa_numeric_table.h"
#include "csr_numeric_table.h"
#include "merged_numeric_table.h"
#include "symmetric_matrix.h"
#include "matrix.h"
#include "data_collection.h"

#include "adaboost_model.h"
#include "adaboost_training_types.h"
#include "adagrad_types.h"
#include "lbfgs_types.h"

#include "apriori_types.h"

#include "brownboost_model.h"
#include "brownboost_training_types.h"

#include "classifier_training_types.h"
#include "classifier_predict_types.h"
#include "binary_confusion_matrix_types.h"
#include "multiclass_confusion_matrix_types.h"

#include "cholesky_types.h"

#include "correlation_distance_types.h"
#include "cosine_distance_types.h"

#include "covariance_types.h"

#include "em_gmm_types.h"

#include "implicit_als_model.h"
#include "implicit_als_training_types.h"
#include "implicit_als_predict_ratings_types.h"
#include "implicit_als_training_init_types.h"

#include "kernel_function_types.h"

#include "kmeans_types.h"
#include "kmeans_init_types.h"

#include "linear_regression_ne_model.h"
#include "linear_regression_qr_model.h"
#include "linear_regression_types.h"

#include "logitboost_model.h"
#include "logitboost_training_types.h"

#include "low_order_moments_types.h"

#include "multi_class_classifier_model.h"
#include "multi_class_classifier_types.h"

#include "multinomial_naive_bayes_model.h"
#include "multinomial_naive_bayes_types.h"
#include "memory_block.h"

#include "outlier_detection_multivariate_types.h"
#include "outlier_detection_univariate_types.h"

#include "pca_types.h"
#include "pivoted_qr_types.h"

#include "qr_types.h"
#include "quantiles_types.h"

#include "ridge_regression_ne_model.h"
#include "ridge_regression_types.h"

#include "stump_model.h"
#include "stump_training_types.h"

#include "svd_types.h"

#include "svm_model.h"
#include "svm_train_types.h"

#include "weak_learner_training_types.h"

#include "sorting_types.h"

#include "relu_types.h"
#include "smoothrelu_types.h"
#include "abs_types.h"
#include "softmax_types.h"
#include "logistic_types.h"
#include "tanh_types.h"

#include "zscore_types.h"

#include "objective_function_types.h"
#include "iterative_solver_types.h"

#include "neural_networks_types.h"
#include "neural_networks_training_model.h"
#include "neural_networks_prediction_model.h"
#include "neural_networks_training_types.h"
#include "neural_networks_prediction_types.h"

#include "layer_forward_types.h"
#include "layer_backward_types.h"

#include "abs/abs_layer_forward_types.h"
#include "abs/abs_layer_backward_types.h"

#include "softmax/softmax_layer_forward_types.h"
#include "softmax/softmax_layer_backward_types.h"

#include "logistic/logistic_layer_forward_types.h"
#include "logistic/logistic_layer_backward_types.h"

#include "relu/relu_layer_forward_types.h"
#include "relu/relu_layer_backward_types.h"

#include "smoothrelu/smoothrelu_layer_forward_types.h"
#include "smoothrelu/smoothrelu_layer_backward_types.h"

#include "tanh/tanh_layer_forward_types.h"
#include "tanh/tanh_layer_backward_types.h"

#include "prelu/prelu_layer_forward_types.h"
#include "prelu/prelu_layer_backward_types.h"

#include "dropout/dropout_layer_forward_types.h"
#include "dropout/dropout_layer_backward_types.h"

#include "batch_normalization/batch_normalization_layer_forward_types.h"
#include "batch_normalization/batch_normalization_layer_backward_types.h"

#include "lrn/lrn_layer_forward_types.h"
#include "lrn/lrn_layer_backward_types.h"

#include "locallyconnected2d/locallyconnected2d_layer_forward_types.h"
#include "locallyconnected2d/locallyconnected2d_layer_backward_types.h"

#include "lcn/lcn_layer_forward_types.h"
#include "lcn/lcn_layer_backward_types.h"

#include "split/split_layer_forward_types.h"
#include "split/split_layer_backward_types.h"

#include "concat/concat_layer_forward_types.h"
#include "concat/concat_layer_backward_types.h"

#include "pooling1d/average_pooling1d_layer_forward_types.h"
#include "pooling1d/average_pooling1d_layer_backward_types.h"

#include "pooling1d/maximum_pooling1d_layer_forward_types.h"
#include "pooling1d/maximum_pooling1d_layer_backward_types.h"

#include "pooling2d/average_pooling2d_layer_forward_types.h"
#include "pooling2d/average_pooling2d_layer_backward_types.h"

#include "pooling2d/maximum_pooling2d_layer_forward_types.h"
#include "pooling2d/maximum_pooling2d_layer_backward_types.h"

#include "pooling2d/stochastic_pooling2d_layer_forward_types.h"
#include "pooling2d/stochastic_pooling2d_layer_backward_types.h"

#include "pooling3d/average_pooling3d_layer_forward_types.h"
#include "pooling3d/average_pooling3d_layer_backward_types.h"

#include "pooling3d/maximum_pooling3d_layer_forward_types.h"
#include "pooling3d/maximum_pooling3d_layer_backward_types.h"

#include "spatial_pooling2d/spatial_average_pooling2d_layer_forward_types.h"
#include "spatial_pooling2d/spatial_average_pooling2d_layer_backward_types.h"

#include "spatial_pooling2d/spatial_maximum_pooling2d_layer_forward_types.h"
#include "spatial_pooling2d/spatial_maximum_pooling2d_layer_backward_types.h"

#include "spatial_pooling2d/spatial_stochastic_pooling2d_layer_forward_types.h"
#include "spatial_pooling2d/spatial_stochastic_pooling2d_layer_backward_types.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
template class BlockDescriptor<int>;
template class BlockDescriptor<float>;
template class BlockDescriptor<double>;
}

#undef __DAAL_CREATOR_ARGUMENTS
#define __DAAL_CREATOR_ARGUMENTS(...) __VA_ARGS__

#undef __DAAL_ADD_TYPE
#define __DAAL_ADD_TYPE(leftPart, rightPart)                    \
    {                                                           \
        registerObject(new leftPart float          rightPart);  \
        registerObject(new leftPart double         rightPart);  \
        registerObject(new leftPart int            rightPart);  \
        registerObject(new leftPart unsigned int   rightPart);  \
        registerObject(new leftPart DAAL_INT64     rightPart);  \
        registerObject(new leftPart DAAL_UINT64    rightPart);  \
        registerObject(new leftPart char           rightPart);  \
        registerObject(new leftPart unsigned char  rightPart);  \
        registerObject(new leftPart short          rightPart);  \
        registerObject(new leftPart unsigned short rightPart);  \
    }

#undef __DAAL_REGISTER_TEMPLATED_OBJECT
#define __DAAL_REGISTER_TEMPLATED_OBJECT(CreatorName, ObjectName, ...)                                                  \
    {                                                                                                                   \
        __DAAL_ADD_TYPE(__DAAL_CREATOR_ARGUMENTS(CreatorName<ObjectName<__VA_ARGS__), __DAAL_CREATOR_ARGUMENTS(> >()))  \
    }

#undef __DAAL_REGISTER_CSR_OBJECT
#define __DAAL_REGISTER_CSR_OBJECT(CreatorName, ObjectName)                                                 \
    {                                                                                                       \
        __DAAL_ADD_TYPE(__DAAL_CREATOR_ARGUMENTS(CreatorName<ObjectName, ), __DAAL_CREATOR_ARGUMENTS(>()))  \
    }

Factory::Factory()
{
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, HomogenNumericTable, );
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, Matrix, );

    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, PackedSymmetricMatrix,  NumericTableIface::upperPackedSymmetricMatrix, );
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, PackedSymmetricMatrix,  NumericTableIface::lowerPackedSymmetricMatrix, );
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, PackedTriangularMatrix, NumericTableIface::upperPackedTriangularMatrix, );
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, PackedTriangularMatrix, NumericTableIface::lowerPackedTriangularMatrix, );

    __DAAL_REGISTER_CSR_OBJECT(CSRCreator, CSRNumericTable);

    registerObject(new Creator<AOSNumericTable>());
    registerObject(new Creator<SOANumericTable>());
    registerObject(new Creator<MergedNumericTable>());
    registerObject(new Creator<NumericTableDictionary>());
    registerObject(new Creator<data_management::DataCollection >());
    registerObject(new Creator<data_management::KeyValueDataCollection >());

    registerObject(new Creator<algorithms::adaboost::Model>());
    registerObject(new Creator<algorithms::adaboost::training::Result>());

    registerObject(new Creator<algorithms::association_rules::Result>());

    registerObject(new Creator<algorithms::brownboost::Model>());
    registerObject(new Creator<algorithms::brownboost::training::Result>());

    registerObject(new Creator<algorithms::cholesky::Result>());

    registerObject(new Creator<algorithms::classifier::training::PartialResult>());
    registerObject(new Creator<algorithms::classifier::quality_metric::binary_confusion_matrix::Result>());
    registerObject(new Creator<algorithms::classifier::quality_metric::multiclass_confusion_matrix::Result>());
    registerObject(new Creator<algorithms::classifier::prediction::Result>());
    registerObject(new Creator<algorithms::classifier::training::Result>());

    registerObject(new Creator<algorithms::correlation_distance::Result>());
    registerObject(new Creator<algorithms::cosine_distance::Result>());

    registerObject(new Creator<algorithms::covariance::PartialResult>());
    registerObject(new Creator<algorithms::covariance::Result>());

    registerObject(new Creator<algorithms::em_gmm::init::Result>());
    registerObject(new Creator<algorithms::em_gmm::Result>());

    registerObject(new Creator<algorithms::implicit_als::Model>());
    registerObject(new Creator<algorithms::implicit_als::PartialModel>());
    registerObject(new Creator<algorithms::implicit_als::prediction::ratings::Result>());
    registerObject(new Creator<algorithms::implicit_als::prediction::ratings::PartialResult>());
    registerObject(new Creator<algorithms::implicit_als::training::init::Result>());
    registerObject(new Creator<algorithms::implicit_als::training::init::PartialResult>());
    registerObject(new Creator<algorithms::implicit_als::training::Result>());
    registerObject(new Creator<algorithms::implicit_als::training::DistributedPartialResultStep1>());
    registerObject(new Creator<algorithms::implicit_als::training::DistributedPartialResultStep2>());
    registerObject(new Creator<algorithms::implicit_als::training::DistributedPartialResultStep3>());
    registerObject(new Creator<algorithms::implicit_als::training::DistributedPartialResultStep4>());

    registerObject(new Creator<algorithms::kernel_function::Result>());

    registerObject(new Creator<algorithms::kmeans::PartialResult>());
    registerObject(new Creator<algorithms::kmeans::Result>());
    registerObject(new Creator<algorithms::kmeans::init::PartialResult>());
    registerObject(new Creator<algorithms::kmeans::init::Result>());

    registerObject(new Creator<algorithms::linear_regression::ModelNormEq>());
    registerObject(new Creator<algorithms::linear_regression::ModelQR    >());
    registerObject(new Creator<algorithms::linear_regression::training::PartialResult>());
    registerObject(new Creator<algorithms::linear_regression::training::Result>());
    registerObject(new Creator<algorithms::linear_regression::prediction::Result>());

    registerObject(new Creator<algorithms::logitboost::Model>());
    registerObject(new Creator<algorithms::logitboost::training::Result>());

    registerObject(new Creator<algorithms::low_order_moments::PartialResult>());
    registerObject(new Creator<algorithms::low_order_moments::Result>());

    registerObject(new Creator<algorithms::multi_class_classifier::Model>());
    registerObject(new Creator<algorithms::multi_class_classifier::training::Result>());

    registerObject(new Creator<algorithms::multinomial_naive_bayes::Model>());
    registerObject(new Creator<algorithms::multinomial_naive_bayes::PartialModel>());
    registerObject(new Creator<algorithms::multinomial_naive_bayes::training::Result>());

    registerObject(new Creator<algorithms::multivariate_outlier_detection::Result>());
    registerObject(new Creator<algorithms::univariate_outlier_detection::Result>());

    registerObject(new Creator<algorithms::pca::Result>());
    registerObject(new Creator<algorithms::pca::PartialResult<algorithms::pca::correlationDense> >());
    registerObject(new Creator<algorithms::pca::PartialResult<algorithms::pca::svdDense        > >());

    registerObject(new Creator<algorithms::pivoted_qr::Result>());

    registerObject(new Creator<algorithms::qr::Result>());
    registerObject(new Creator<algorithms::qr::OnlinePartialResult>());
    registerObject(new Creator<algorithms::qr::DistributedPartialResult>());
    registerObject(new Creator<algorithms::qr::DistributedPartialResultStep3>());

    registerObject(new Creator<algorithms::quantiles::Result>());

    registerObject(new Creator<algorithms::math::relu::Result>());
    registerObject(new Creator<algorithms::math::softmax::Result>());
    registerObject(new Creator<algorithms::math::logistic::Result>());
    registerObject(new Creator<algorithms::math::smoothrelu::Result>());
    registerObject(new Creator<algorithms::math::abs::Result>());
    registerObject(new Creator<algorithms::math::tanh::Result>());

    registerObject(new Creator<algorithms::ridge_regression::ModelNormEq>());
    registerObject(new Creator<algorithms::ridge_regression::training::PartialResult>());
    registerObject(new Creator<algorithms::ridge_regression::training::Result>());
    registerObject(new Creator<algorithms::ridge_regression::prediction::Result>());

    registerObject(new Creator<algorithms::stump::Model>());
    registerObject(new Creator<algorithms::stump::training::Result>());

    registerObject(new Creator<algorithms::svd::Result>());
    registerObject(new Creator<algorithms::svd::OnlinePartialResult>());
    registerObject(new Creator<algorithms::svd::DistributedPartialResult>());
    registerObject(new Creator<algorithms::svd::DistributedPartialResultStep3>());

    registerObject(new Creator<algorithms::svm::Model>());
    registerObject(new Creator<algorithms::svm::training::Result>());

    registerObject(new Creator<algorithms::weak_learner::training::Result>());

    registerObject(new Creator<algorithms::sorting::Result>());

    registerObject(new Creator<algorithms::normalization::zscore::Result>());

    registerObject(new Creator<algorithms::optimization_solver::objective_function::Result>());
    registerObject(new Creator<algorithms::optimization_solver::iterative_solver::Result>());
    registerObject(new Creator<algorithms::optimization_solver::adagrad::Result>());
    registerObject(new Creator<algorithms::optimization_solver::lbfgs::Result>());

    registerObject(new Creator<algorithms::neural_networks::training::Result>());
    registerObject(new Creator<algorithms::neural_networks::prediction::Result>());

    registerObject(new Creator<algorithms::neural_networks::training::Model>());
    registerObject(new Creator<algorithms::neural_networks::prediction::Model>());

    registerObject(new Creator<algorithms::neural_networks::layers::abs::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::abs::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::softmax::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::softmax::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::logistic::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::logistic::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::relu::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::relu::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::smoothrelu::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::smoothrelu::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::tanh::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::tanh::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::prelu::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::prelu::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::dropout::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::dropout::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::batch_normalization::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::batch_normalization::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::lrn::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::lrn::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::locallyconnected2d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::locallyconnected2d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::lcn::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::lcn::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::split::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::split::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::concat::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::concat::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::average_pooling1d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::average_pooling1d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::maximum_pooling1d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::maximum_pooling1d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::average_pooling2d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::average_pooling2d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::maximum_pooling2d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::maximum_pooling2d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::stochastic_pooling2d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::stochastic_pooling2d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::average_pooling3d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::average_pooling3d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::maximum_pooling3d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::maximum_pooling3d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::spatial_average_pooling2d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::spatial_average_pooling2d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::spatial_maximum_pooling2d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::spatial_maximum_pooling2d::backward::Result>());

    registerObject(new Creator<algorithms::neural_networks::layers::spatial_stochastic_pooling2d::forward::Result>());
    registerObject(new Creator<algorithms::neural_networks::layers::spatial_stochastic_pooling2d::backward::Result>());

    registerObject(new Creator<algorithms::OptionalArgument >());
    registerObject(new Creator<data_management::MemoryBlock >());
}

Factory::~Factory()
{
    for(int i = 0; i < _map.size(); i++)
    {
        delete _map[i]->second();
    }
}

void Factory::registerObject(AbstractCreator *creator)
{
    _map << services::SharedPtr<AbstractCreatorPair>(new AbstractCreatorPair(creator->getTag(), creator));
}

Factory &Factory::instance()
{
    static Factory obj;
    return obj;
}

SerializationIface *Factory::createObject(int objectId)
{
    int pos = _map.find(objectId);
    if(pos == -1)
    {
        return NULL;
    }
    return _map[pos]->second()->create();
}

Factory::Factory(const Factory &) {}
Factory &Factory::operator = (const Factory &factory) { return (*this); }

void SerializationIface::serialize(InputDataArchive &archive)
{
    archive.segmentHeader( getSerializationTag() );
    serializeImpl( &archive );
    archive.segmentFooter();
}

void SerializationIface::deserialize(OutputDataArchive &archive)
{
    int tag = archive.segmentHeader();
    deserializeImpl( &archive );
    archive.segmentFooter();
}

} // namespace data_management

algorithms::Argument::Argument(const size_t n) : _storage(new data_management::DataCollection(n)), idx(0) {}

const data_management::SerializationIfacePtr& algorithms::Argument::get(size_t index) const
{
    return (*_storage)[index];
}

/**
* Sets the element to the specified position in the Argument
* \param[in] index Index of the element
* \param[in] value Pointer to the element
* \return Reference to the requested element
*/
void algorithms::Argument::set(size_t index, const data_management::SerializationIfacePtr &value)
{
    (*_storage)[index] = value;
    return;
}

} // namespace daal
