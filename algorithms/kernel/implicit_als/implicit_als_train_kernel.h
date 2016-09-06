/* file: implicit_als_train_kernel.h */
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
//  Declaration of structure containing kernels for implicit ALS
//  training.
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_KERNEL_H__
#define __IMPLICIT_ALS_TRAIN_KERNEL_H__

#include "implicit_als_training_batch.h"
#include "implicit_als_model.h"
#include "kernel.h"
#include "threading.h"

#include "service_micro_table.h"
#include "service_memory.h"

using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainTaskBase;

template <typename algorithmFPType, Method method, CpuType cpu>
struct ImplicitALSTrainTask;

template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainDistrStep3Task;

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainKernelCommon : public daal::algorithms::Kernel
{
protected:
    void computeXtX(size_t *nRows, size_t *nCols, algorithmFPType *beta, algorithmFPType *x, size_t *ldx,
                algorithmFPType *xtx, size_t *ldxtx);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainKernelBase : public ImplicitALSTrainKernelCommon<algorithmFPType, cpu>
{
protected:
    friend struct ImplicitALSTrainTaskBase<algorithmFPType, cpu>;
    friend struct ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu>;
    friend struct ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu>;
    friend struct ImplicitALSTrainDistrStep3Task<algorithmFPType, cpu>;

    void updateSystem(size_t *nCols, algorithmFPType *x, algorithmFPType *coeff, algorithmFPType *p,
                algorithmFPType *a, size_t *lda, algorithmFPType *b);

    void solve(size_t *nCols, algorithmFPType *a, size_t *lda, algorithmFPType *b, size_t *ldb);

    void computeFactors(
                size_t nRows, size_t nCols, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
                size_t nFactors, algorithmFPType *colFactors, algorithmFPType *rowFactors,
                algorithmFPType alpha, algorithmFPType lambda, algorithmFPType *xtx, daal::tls<algorithmFPType *> *lhs);

    virtual void formSystem(size_t i, size_t nCols, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
                size_t nFactors, algorithmFPType *colFactors,
                algorithmFPType alpha, algorithmFPType *lhs, algorithmFPType *rhs, algorithmFPType lambda) = 0;

    virtual void computeCostFunction(size_t nItems, size_t nUsers, size_t nFactors, algorithmFPType *data,
                size_t *colIndices, size_t *rowOffsets, algorithmFPType *itemsFactors, algorithmFPType *usersFactors,
                algorithmFPType alpha, algorithmFPType lambda, algorithmFPType *costFunctionPtr) = 0;
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSTrainKernel : public ImplicitALSTrainKernelBase<algorithmFPType, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSTrainKernelBase<algorithmFPType, cpu>
{
protected:
    void formSystem(size_t i, size_t nCols, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
                size_t nFactors, algorithmFPType *colFactors,
                algorithmFPType alpha, algorithmFPType *lhs, algorithmFPType *rhs, algorithmFPType lambda);

    void computeCostFunction(size_t nItems, size_t nUsers, size_t nFactors, algorithmFPType *data,
                size_t *colIndices, size_t *rowOffsets, algorithmFPType *itemsFactors, algorithmFPType *usersFactors,
                algorithmFPType alpha, algorithmFPType lambda, algorithmFPType *costFunctionPtr);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainKernel<algorithmFPType, defaultDense, cpu> : public ImplicitALSTrainKernelBase<algorithmFPType, cpu>
{
protected:
    void formSystem(size_t i, size_t nCols, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
                size_t nFactors, algorithmFPType *colFactors,
                algorithmFPType alpha, algorithmFPType *lhs, algorithmFPType *rhs, algorithmFPType lambda);

    void computeCostFunction(size_t nItems, size_t nUsers, size_t nFactors, algorithmFPType *data,
                size_t *colIndices, size_t *rowOffsets, algorithmFPType *itemsFactors, algorithmFPType *usersFactors,
                algorithmFPType alpha, algorithmFPType lambda, algorithmFPType *costFunctionPtr);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSTrainBatchKernel : public ImplicitALSTrainKernel<algorithmFPType, method, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainBatchKernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu>
{
public:
    void compute(const NumericTable *data, implicit_als::Model *initModel, implicit_als::Model *model,
                const Parameter *parameter);

    friend struct ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu>;
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainBatchKernel<algorithmFPType, defaultDense, cpu> : public ImplicitALSTrainKernel<algorithmFPType, defaultDense, cpu>
{
public:
    void compute(const NumericTable *data, implicit_als::Model *initModel, implicit_als::Model *model,
                const Parameter *parameter);

    friend struct ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu>;
};


template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainTaskBase
{
    ImplicitALSTrainTaskBase(const NumericTable *dataTable, implicit_als::Model *initModel, implicit_als::Model *model, const Parameter *parameter,
                ImplicitALSTrainKernelBase<algorithmFPType, cpu> *algorithm);

    virtual ~ImplicitALSTrainTaskBase();

    void getData();

    size_t nItems;
    size_t nUsers;
    size_t nFactors;

    daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu> mtItemsFactors;
    daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu> mtUsersFactors;

    algorithmFPType *itemsFactors;
    algorithmFPType *usersFactors;

    algorithmFPType *xtx;
    daal::tls<algorithmFPType *> *lhs;

protected:
    ImplicitALSTrainKernelBase<algorithmFPType, cpu> *_algorithm;
};

template <typename algorithmFPType, Method method, CpuType cpu>
struct ImplicitALSTrainTask : ImplicitALSTrainTaskBase<algorithmFPType, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu> : ImplicitALSTrainTaskBase<algorithmFPType, cpu>
{
    ImplicitALSTrainTask(const NumericTable *dataTable, implicit_als::Model *initModel, implicit_als::Model *model, const Parameter *parameter,
                ImplicitALSTrainKernelBase<algorithmFPType, cpu> *algorithm);

    virtual ~ImplicitALSTrainTask();

    void getData();

    void csr2csc(size_t nItems, size_t nUsers,
                algorithmFPType *csrdata, size_t *colIndices, size_t *rowOffsets,
                algorithmFPType *cscdata, size_t *rowIndices, size_t *colOffsets);

    static void *operator new(std::size_t sz)
    {
        return daal::services::daal_malloc(sz);
    }

    static void operator delete(void *ptr, std::size_t sz)
    {
        daal::services::daal_free(ptr);
    }

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nItems;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nUsers;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nFactors;

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::mtItemsFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::mtUsersFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::itemsFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::usersFactors;

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::xtx;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::lhs;

    daal::internal::CSRBlockMicroTable<algorithmFPType, readOnly, cpu> mtData;

    algorithmFPType *data;
    size_t *colIndices;
    size_t *rowOffsets;

    algorithmFPType *tdata;
    size_t *rowIndices;
    size_t *colOffsets;
protected:
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::_algorithm;
};

template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu> : ImplicitALSTrainTaskBase<algorithmFPType, cpu>
{
    ImplicitALSTrainTask(const NumericTable *dataTable, implicit_als::Model *initModel, implicit_als::Model *model, const Parameter *parameter,
                ImplicitALSTrainKernelBase<algorithmFPType, cpu> *algorithm);

    virtual ~ImplicitALSTrainTask();

    void getData();

    void transpose(size_t nItems, size_t nUsers, algorithmFPType *data, algorithmFPType *tdata);

    static void *operator new(std::size_t sz)
    {
        return daal::services::daal_malloc(sz);
    }

    static void operator delete(void *ptr, std::size_t sz)
    {
        daal::services::daal_free(ptr);
    }

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nItems;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nUsers;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nFactors;

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::mtItemsFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::mtUsersFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::itemsFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::usersFactors;

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::xtx;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::lhs;

    daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> mtData;

    algorithmFPType *data;
    algorithmFPType *tdata;

protected:
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::_algorithm;
};

template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainDistrStep3Task
{
    ImplicitALSTrainDistrStep3Task(implicit_als::PartialModel *srcModel, implicit_als::PartialModel *dstModel,
                const Parameter *parameter, ImplicitALSTrainKernelBase<algorithmFPType, cpu> *algorithm);

    virtual ~ImplicitALSTrainDistrStep3Task();

    size_t nSrcRows;
    size_t nDstRows;
    size_t nFactors;

    algorithmFPType *srcFactors;
    algorithmFPType *dstFactors;
    algorithmFPType *xtx;
    daal::tls<algorithmFPType *> *lhs;

protected:
    daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> mtXTX;
    daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> mtSrcFactors;
    daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu> mtDstFactors;
    ImplicitALSTrainKernelBase<algorithmFPType, cpu> *_algorithm;
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainDistrStep1Kernel : public daal::algorithms::Kernel
{
public:
    void compute(implicit_als::PartialModel *partialModel, data_management::NumericTable *crossProduct,
                const Parameter *parameter);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainDistrStep2Kernel : public daal::algorithms::Kernel
{
public:
    void compute(size_t nParts, data_management::NumericTable **partialCrossProducts,
                data_management::NumericTable *crossProduct, const Parameter *parameter);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainDistrStep3Kernel : public ImplicitALSTrainKernelCommon<algorithmFPType, cpu>
{
public:
    void compute(implicit_als::PartialModel *partialModel, data_management::NumericTable *offsetTable,
                data_management::KeyValueDataCollection *models, const Parameter *parameter);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSTrainDistrStep4Kernel : public ImplicitALSTrainKernel<algorithmFPType, method, cpu>
{
public:
    void compute(data_management::KeyValueDataCollection *models, data_management::NumericTable *dataTable,
                data_management::NumericTable *cpTable, implicit_als::PartialModel *partialModel,
                const Parameter *parameter);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainDistrStep4Kernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu>
{
public:
    void compute(data_management::KeyValueDataCollection *models, data_management::NumericTable *dataTable,
                data_management::NumericTable *cpTable, implicit_als::PartialModel *partialModel,
                const Parameter *parameter);

protected:
    void formSystem(size_t i, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
                size_t nFactors, size_t nBlocks, size_t *nColFactorsRows,
                int **indices,
                daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> **mtFactors,
                algorithmFPType alpha, algorithmFPType *lhs, algorithmFPType *rhs, algorithmFPType lambda,
                services::SharedPtr<services::Error> &error);
};

}
}
}
}
}

#endif
