/* file: kdtree_knn_classification_model_impl.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the class defining the K-Nearest Neighbors (kNN) model
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_MODEL_IMPL_
#define __KDTREE_KNN_CLASSIFICATION_MODEL_IMPL_

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace interface1
{

struct KDTreeNode
{
    size_t dimension;
    size_t leftIndex;
    size_t rightIndex;
    double cutPoint;
};

class KDTreeTable : public data_management::AOSNumericTable
{
public:
    KDTreeTable(size_t rowCount = 0) : data_management::AOSNumericTable(sizeof(KDTreeNode), 4, rowCount)
    {
        setFeature<size_t> (0, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, dimension));
        setFeature<size_t> (1, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, leftIndex));
        setFeature<size_t> (2, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, rightIndex));
        setFeature<double> (3, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, cutPoint));
        allocateDataMemory();
    }
};

class Model::ModelImpl
{
public:
    /**
     * Empty constructor for deserialization
     */
    ModelImpl() : _kdTreeTable(), _rootNodeIndex(0), _lastNodeIndex(0), _data(), _labels() {}

    /**
     * Returns the KD-tree table
     * \return KD-tree table
     */
    services::SharedPtr<KDTreeTable> getKDTreeTable() { return _kdTreeTable; }

    /**
     * Returns the KD-tree table
     * \return KD-tree table
     */
    services::SharedPtr<const KDTreeTable> getKDTreeTable() const { return _kdTreeTable; }

    /**
    *  Sets a KD-tree table
    *  \param[in]  value  KD-tree table
    */
    void setKDTreeTable(const services::SharedPtr<KDTreeTable> & value) { _kdTreeTable = value; }

    /**
     * Returns the index of KD-tree root node
     * \return Index of KD-tree root node
     */
    size_t getRootNodeIndex() const { return _rootNodeIndex; }

    /**
    *  Sets a index of KD-tree root node
    *  \param[in]  value  Index of KD-tree root node
    */
    void setRootNodeIndex(size_t value) { _rootNodeIndex = value; }

    /**
     * Returns the index of first part KD-tree last node
     * \return Index of first part KD-tree last node
     */
    size_t getLastNodeIndex() const { return _lastNodeIndex; }

    /**
    *  Sets a index of first part KD-tree last node
    *  \param[in]  value  Index of first part KD-tree last node
    */
    void setLastNodeIndex(size_t value) { _lastNodeIndex = value; }

    /**
     * Returns training data
     * \return Training data
     */
    data_management::NumericTableConstPtr getData() const { return _data; }

    /**
     * Returns training data
     * \return Training data
     */
    data_management::NumericTablePtr getData() { return _data; }

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive * arch)
    {
        arch->set(_rootNodeIndex);
        arch->set(_lastNodeIndex);
        arch->setSharedPtrObj(_kdTreeTable);
        arch->setSharedPtrObj(_data);
        arch->setSharedPtrObj(_labels);
    }

    /**
    *  Sets a training data
    *  \param[in]  value  Training data
    *  \param[in]  copy   Flag indicating necessary of data deep copying to avoid direct usage and modification of input data.
    */
    template <typename algorithmFPType>
    DAAL_EXPORT void setData(const data_management::NumericTablePtr & value, bool copy)
    {
        if (!copy)
        {
            _data = value;
        }
        else
        {
            services::SharedPtr<data_management::SOANumericTable> tbl(new data_management::SOANumericTable(value->getNumberOfColumns(),
                                                                                                           value->getNumberOfRows(),
                                                                                                           data_management::DictionaryIface::equal));
            tbl->getDictionary()->setAllFeatures<algorithmFPType>(); // Just to set type of all features. Also, no way to use featuresEqual flag.
            tbl->resize(value->getNumberOfRows());

            tbl->allocateDataMemory();
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            tbl->getBlockOfRows(0, tbl->getNumberOfRows(), data_management::writeOnly, destBD);
            value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD);
            services::daal_memcpy_s(destBD.getBlockPtr(), destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFPType),
                                    srcBD.getBlockPtr(), srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFPType));
            tbl->releaseBlockOfRows(destBD);
            value->releaseBlockOfRows(srcBD);
            _data = tbl;
        }
    }

    /**
     * Returns training labels
     * \return Training labels
     */
    services::SharedPtr<const data_management::NumericTable> getLabels() const { return _labels; }

    /**
     * Returns training labels
     * \return Training labels
     */
    services::SharedPtr<data_management::NumericTable> getLabels() { return _labels; }

    /**
    *  Sets a training data
    *  \param[in]  value  Training labels
    *  \param[in]  copy   Flag indicating necessary of data deep copying to avoid direct usage and modification of input labels.
    */
    template <typename algorithmFPType>
    DAAL_EXPORT void setLabels(const data_management::NumericTablePtr & value, bool copy)
    {
        if (!copy)
        {
            _labels = value;
        }
        else
        {
            services::SharedPtr<data_management::SOANumericTable> tbl(new data_management::SOANumericTable(value->getNumberOfColumns(),
                                                                                                           value->getNumberOfRows()));
            tbl->setArray(static_cast<algorithmFPType *>(0), 0); // Just to create the dictionary.
            tbl->getDictionary()->setNumberOfFeatures(value->getNumberOfColumns()); // Sadly, setArray() hides number of features from the dictionary.
            data_management::NumericTableFeature temp;
            temp.setType<algorithmFPType>();
            tbl->getDictionary()->setAllFeatures(temp); // Just to set type of all features. Also, no way to use featuresEqual flag.
            tbl->allocateDataMemory();
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            tbl->getBlockOfRows(0, tbl->getNumberOfRows(), data_management::writeOnly, destBD);
            value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD);
            services::daal_memcpy_s(destBD.getBlockPtr(), destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFPType),
                                    srcBD.getBlockPtr(), srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFPType));
            tbl->releaseBlockOfRows(destBD);
            value->releaseBlockOfRows(srcBD);
            _labels = tbl;
        }
    }

private:
    services::SharedPtr<KDTreeTable> _kdTreeTable;
    size_t _rootNodeIndex;
    size_t _lastNodeIndex;
    data_management::NumericTablePtr _data;
    data_management::NumericTablePtr _labels;
};

} // namespace interface1

using interface1::KDTreeTable;
using interface1::KDTreeNode;

} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
