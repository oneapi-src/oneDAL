/* file: df_model_impl.h */
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
//  Implementation of the class defining the decision forest model
//--
*/

#ifndef __DF_MODEL_IMPL__
#define __DF_MODEL_IMPL__

#include "env_detect.h"
#include "daal_shared_ptr.h"
#include "service_defines.h"

typedef size_t ClassIndexType;
typedef double ClassificationFPType; //type of features stored in classification model
typedef double RegressionFPType; //type of features and regression response stored in the model

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace internal
{

//Simple container
template<typename T, int dummy>
class TVector
{
public:
    TVector(size_t n = 0) : _data(nullptr), _size(0){ if(n) alloc(n); }
    TVector(size_t n, T val) : _data(nullptr), _size(0)
    {
        if(n)
        {
            alloc(n);
            for(size_t i = 0; i < n; ++i)
                _data[i] = val;
        }
    }
    ~TVector() { destroy(); }
    TVector(const TVector& o) : _data(nullptr), _size(0)
    {
        if(o._size)
        {
            alloc(o._size);
            daal::services::daal_memcpy_s(_data, sizeof(T)*_size, o._data, sizeof(T)*_size);
        }
    }

    TVector& operator=(const TVector& o)
    {
        if(this != &o)
        {
            if(_size < o._size)
            {
                destroy();
                alloc(o._size);
            }
            daal::services::daal_memcpy_s(_data, sizeof(T)*_size, o._data, sizeof(T)*_size);
        }
        return *this;
    }

    size_t size() const { return _size; }

    void resize(size_t n, T val)
    {
        if(n != _size)
        {
            destroy();
            alloc(n);
        }
        for(size_t i = 0; i < _size; ++i)
            _data[i] = val;
    }

    T &operator [] (size_t index)
    {
        DAAL_ASSERT(index < size());
        return _data[index];
    }

    const T &operator [] (size_t index) const
    {
        DAAL_ASSERT(index < size());
        return _data[index];
    }
    T* detach() { auto res = _data; _data = nullptr; _size = 0;  return res; }
    T* get() { return _data; }
    const T* get() const { return _data; }

private:
    void alloc(size_t n)
    {
        _data = (T*)(n ? daal::services::daal_malloc(n*sizeof(T)) : nullptr);
        if(_data)
            _size = n;
    }

    void destroy()
    {
        if(_data)
        {
            daal::services::daal_free(_data);
            _data = nullptr;
            _size = 0;
        }
    }

private:
    T* _data;
    size_t _size;
};

template <typename TResponse, typename THistogramm>
struct ClassifierResponse
{
    TResponse    value; //majority votes response

#ifdef KEEP_CLASSES_PROBABILITIIES
    THistogramm* hist;  //histogramm
    size_t       size;  //number of classes in histogramm
    ClassifierResponse() : hist(nullptr), value(0){}
    ~ClassifierResponse() { if(hist) daal::services::daal_free(hist); }
#else
    ClassifierResponse() : value(0){}
#endif
    ClassifierResponse(const ClassifierResponse&) = delete;
    ClassifierResponse& operator=(const ClassifierResponse& o) = delete;
};

struct TreeNodeBase
{
    virtual ~TreeNodeBase(){}
    virtual bool isSplit() const = 0;

    TreeNodeBase() : count(0), impurity(0){}
    size_t count;
    double impurity;
};

template <typename algorithmFPType>
struct TreeNodeSplit : public TreeNodeBase
{
    typedef algorithmFPType FeatureType;
    FeatureType featureValue;
    TreeNodeBase* kid[2];
    int featureIdx;
    bool featureUnordered;

    TreeNodeSplit() { kid[0] = kid[1] = nullptr; }
    const TreeNodeBase* left() const { return kid[0]; }
    const TreeNodeBase* right() const { return kid[1]; }
    TreeNodeBase* left()  { return kid[0]; }
    TreeNodeBase* right() { return kid[1]; }

    void set(int featIdx, algorithmFPType featValue, bool bUnordered) { featureValue = featValue; featureIdx = featIdx; featureUnordered = bUnordered; }
    virtual bool isSplit() const { return true; }
};

template <typename TResponseType>
struct TreeNodeLeaf: public TreeNodeBase
{
    TResponseType response;

    TreeNodeLeaf(){}
    virtual bool isSplit() const { return false; }
};

template<typename algorithmFPType>
struct TreeNodeRegression
{
    typedef TreeNodeBase Base;
    typedef TreeNodeSplit<algorithmFPType> Split;
    typedef TreeNodeLeaf<algorithmFPType> Leaf;

    static Leaf* castLeaf(Base* n) { return static_cast<Leaf*>(n); }
    static const Leaf* castLeaf(const Base* n) { return static_cast<const Leaf*>(n); }
    static Split* castSplit(Base* n) { return static_cast<Split*>(n); }
    static const Split* castSplit(const Base* n) { return static_cast<const Split*>(n); }
};

template<typename algorithmFPType>
struct TreeNodeClassification
{
    typedef TreeNodeBase Base;
    typedef TreeNodeSplit<algorithmFPType> Split;
    typedef TreeNodeLeaf<ClassifierResponse<ClassIndexType, size_t>> Leaf;

    static Leaf* castLeaf(Base* n) { return static_cast<Leaf*>(n); }
    static const Leaf* castLeaf(const Base* n) { return static_cast<const Leaf*>(n); }
    static Split* castSplit(Base* n) { return static_cast<Split*>(n); }
    static const Split* castSplit(const Base* n) { return static_cast<const Split*>(n); }
};

template <typename NodeType>
class HeapMemoryAllocator
{
public:
    static typename NodeType::Leaf* allocLeaf();
    static typename NodeType::Split* allocSplit();
    static void free(typename NodeType::Base* n);
};

template <typename NodeType, typename Allocator>
void deleteNode(typename NodeType::Base* n)
{
    if(n->isSplit())
    {
        typename NodeType::Split* s = static_cast<typename NodeType::Split*>(n);
        if(s->left())
            deleteNode<NodeType, Allocator>(s->left());
        if(s->right())
            deleteNode<NodeType, Allocator>(s->right());
    }
    Allocator::free(n);
}

class Tree : public Base
{
public:
    Tree(){}
    virtual ~Tree();
};
typedef services::SharedPtr<Tree> TreePtr;

template <typename TNodeType, typename TAllocator = HeapMemoryAllocator<TNodeType> >
class TreeImpl : public Tree
{
public:
    typedef TAllocator Allocator;
    typedef TNodeType NodeType;
    typedef TreeImpl<TNodeType, TAllocator> ThisType;

    TreeImpl(typename NodeType::Base* t, bool bHasUnorderedFeatureSplits) : _top(t), _hasUnorderedFeatureSplits(bHasUnorderedFeatureSplits){}
    ~TreeImpl();
    const typename NodeType::Base* top() const { return _top; }
    bool hasUnorderedFeatureSplits() const { return _hasUnorderedFeatureSplits; }

private:
    typename NodeType::Base* _top;
    bool _hasUnorderedFeatureSplits;
};
template<typename Allocator = HeapMemoryAllocator<TreeNodeRegression<RegressionFPType> > >
using TreeImpRegression = TreeImpl<TreeNodeRegression<RegressionFPType>, Allocator>;

template<typename Allocator = HeapMemoryAllocator<TreeNodeClassification<ClassificationFPType> > >
using TreeImpClassification = TreeImpl<TreeNodeClassification<ClassificationFPType>, Allocator>;

class ModelImpl
{
public:
    ModelImpl();
    ~ModelImpl();

    size_t size() const { return _nTree.get(); }
    const Tree* at(size_t i) const { return _aTree[i];  }
    bool add(Tree* pTree);
    bool reserve(size_t nTrees);

protected:
    void destroy();

protected:
    Tree** _aTree;
    daal::services::Atomic<size_t> _nTree;
    size_t _nCapacity;
};

template <typename NodeType>
typename NodeType::Leaf* HeapMemoryAllocator<NodeType>::allocLeaf() { return new typename NodeType::Leaf(); }

template <typename NodeType>
typename NodeType::Split* HeapMemoryAllocator<NodeType>::allocSplit() { return new typename NodeType::Split(); }

template <typename NodeType>
void HeapMemoryAllocator<NodeType>::free(typename NodeType::Base* n) { delete n; }

template <typename NodeType, typename Allocator>
TreeImpl<NodeType, Allocator>::~TreeImpl()
{
    if(_top)
        deleteNode<NodeType, Allocator>(_top);
}

} // namespace internal
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#define __DAAL_REGISTER_SERIALIZATION_CLASS2(ClassName, ImplClassName, Tag)\
    static data_management::SerializationIface* creator##ClassName() { return new ImplClassName(); }\
    data_management::SerializationDesc ClassName::_desc(creator##ClassName, Tag); \
    int ClassName::serializationTag() { return _desc.tag(); }\
    int ClassName::getSerializationTag() const { return _desc.tag(); }

#endif
