/* file: decision_stump_train_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

template <typename SplitCriterion, typename LeavesData>
void trainStump(SplitCriterion &splitCriterion, LeavesData &leavesData, const NumericTable &x, const NumericTable &y, const NumericTable *w,
                size_t numberOfClasses = 0, size_t minLeafObservations = 1, size_t minSplitObservations = 2)
{
    const size_t xRowCount = x.getNumberOfRows();
    const size_t xColumnCount = x.getNumberOfColumns();
    DAAL_ASSERT(xRowCount > 0);
    DAAL_ASSERT(xColumnCount > 0);
    DAAL_ASSERT(xRowCount == y.getNumberOfRows());

    FeatureTypesCache featureTypesCache(x);

    typename SplitCriterion::DataStatistics totalDataStatistics(numberOfClasses, x, y, w), dataStatistics(numberOfClasses, w);

    size_t * indexes = prepareIndexes(xRowCount);

    clear();

    TreeNodeIndex nodeIndex = pushBack();

    BlockDescriptor<IndependentVariableType> * xBD = new BlockDescriptor<IndependentVariableType>[xColumnCount];
    const IndependentVariableType ** dx = new const IndependentVariableType * [xColumnCount];
    BlockDescriptor<DependentVariableType> yBD;
    const_cast<NumericTable *>(&y)->getBlockOfColumnValues(0, 0, xRowCount, readOnly, yBD);

    BlockDescriptor<IndependentVariableType> wBD;
    if (w) { const_cast<NumericTable *>(w)->getBlockOfColumnValues(0, 0, xRowCount, readOnly, wBD);}

    const TrainigContext<SplitCriterion, LeavesData> context{ splitCriterion, leavesData, x, y, w, featureTypesCache, dataStatistics,
              minLeafObservations, minSplitObservations, dx, yBD.getBlockPtr(),
              wBD.getBlockPtr() };


    if (xRowCount < context.minSplitSize || xRowCount < context.minLeafSize * 2)
    {
        makeLeaf(nodeIndex, totalDataStatistics.getBestDependentVariableValue(), context.leavesData.add(totalDataStatistics),
                 static_cast<double>(context.splitCriterion(totalDataStatistics, xRowCount)), static_cast<int>(xRowCount));
        return;
    }

    {
        typename SplitCriterion::DependentVariableType leafDependentVariableValue;
        if (totalDataStatistics.isPure(leafDependentVariableValue))
        {
            makeLeaf(nodeIndex, leafDependentVariableValue, context.leavesData.add(totalDataStatistics),
                     static_cast<double>(context.splitCriterion(totalDataStatistics, xRowCount)), static_cast<int>(xRowCount));
            return;
        }
    }

    FeatureIndex winnerFeatureIndex = 0;
    IndependentVariableType winnerCutPoint;
    typename SplitCriterion::ValueType winnerSplitCriterionValue;
    size_t winnerPointsAtLeft;
    typename SplitCriterion::DataStatistics winnerDataStatistics;
    bool winnerIsLeaf = true;

    struct Item
    {
        IndependentVariable x;
        IndependentVariable w;
        DependentVariable y;
    };

    struct Local
    {
        FeatureIndex winnerFeatureIndex;
        IndependentVariableType winnerCutPoint;
        typename SplitCriterion::ValueType winnerSplitCriterionValue, splitCriterionValue;
        size_t winnerPointsAtLeft;
        typename SplitCriterion::DataStatistics winnerDataStatistics, bestCutPointDataStatistics, dataStatistics;
        bool winnerIsLeaf;
        SplitCriterion splitCriterion;
        Item* items;

        Local(const SplitCriterion &criterion, const size_t nRows) : winnerIsLeaf(true), splitCriterion(criterion)
        {
            items = daal_alloc<Item>(nRows);
        }

        ~Local()
        {
            daal_free(items);
            items = nullptr;
        }
    };

    daal::tls<Local *> localTLS([ =, &context]()-> Local *
    {
        Local *const ptr = new Local(context.splitCriterion, xRowCount);
        return ptr;
    } );

    typedef daal::internal::Math<typename SplitCriterion::ValueType, cpu> SplitCriterionMath;
    typedef daal::services::internal::EpsilonVal<typename SplitCriterion::ValueType> SplitCriterionEpsilon;
    const typename SplitCriterion::ValueType epsilon = SplitCriterionEpsilon::get();

    daal::threader_for(xColumnCount, xColumnCount, [ =, &context, &localTLS, &totalDataStatistics](size_t featureIndex)
    {
        const_cast<NumericTable *>(&context.x)->getBlockOfColumnValues(featureIndex, 0, xRowCount, readOnly, xBD[featureIndex]);
        dx[featureIndex] = xBD[featureIndex].getBlockPtr();

        Local *const local = localTLS.local();

        Item *const items = local->items;

        const size_t rowsPerBlock = 512;
        const size_t blockCount = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;
        for(size_t iBlock = 0; iBlock < blockCount; iBlock++)
        {
            const size_t first = iBlock * rowsPerBlock;
            const size_t last = min<cpu>(first + rowsPerBlock, xRowCount);

            for (size_t i = first; i < last; ++i)
            {
                items[i].x = context.dx[featureIndex][indexes[i]];
                items[i].y = context.dy[indexes[i]];
            }
            if (context.dw)
            {
                for (size_t i = first; i < last; ++i)
                {
                    items[i].w = context.dw[indexes[i]];
                }
            }
        }

        introSort<cpu>(items, &items[xRowCount], [](const Item & v1, const Item & v2) -> bool
        {
            return v1.x < v2.x;
        });
        DAAL_ASSERT(isSorted<cpu>(items, &items[xRowCount], [](const Item & v1, const Item & v2) -> bool
        {
            return v1.x < v2.x;
        }));

        Item *next = nullptr;
        const auto i = CutPointFinder<cpu, IndependentVariableType, SplitCriterion>::find(local->splitCriterion, items, &items[xRowCount],
                       local->dataStatistics,
                       totalDataStatistics,
                       context.featureTypesCache[featureIndex], next, local->splitCriterionValue,
                       local->bestCutPointDataStatistics,
                       [](const Item & v) -> IndependentVariableType { return v.x; },
                       [](const Item & v) -> DependentVariable { return v.y; },
                       [](const Item & v) -> IndependentVariableType { return v.w; },
                       [](const Item & v1, const Item & v2) -> bool { return v1.x < v2.x; });

        if (i != &items[xRowCount] && (local->winnerIsLeaf || local->splitCriterionValue < local->winnerSplitCriterionValue ||
                                       (SplitCriterionMath::sFabs(local->splitCriterionValue - local->winnerSplitCriterionValue) <= epsilon &&
                                        local->winnerFeatureIndex > featureIndex)))
        {
            local->winnerIsLeaf = false;
            local->winnerFeatureIndex = featureIndex;
            local->winnerSplitCriterionValue = local->splitCriterionValue;
            switch (context.featureTypesCache[featureIndex])
            {
            case data_management::features::DAAL_CATEGORICAL:
                local->winnerCutPoint = i->x;
                break;
            case data_management::features::DAAL_ORDINAL:
                local->winnerCutPoint = next->x;
                break;
            case data_management::features::DAAL_CONTINUOUS:
                local->winnerCutPoint = (i->x + next->x) / 2;
                break;
            default:
                DAAL_ASSERT(false);
                break;
            }
            local->winnerPointsAtLeft = next - items; // distance.
            local->winnerDataStatistics = local->bestCutPointDataStatistics;
        }
        // daal_free(items);
    });

    localTLS.reduce([ =, &winnerIsLeaf, &winnerSplitCriterionValue, &winnerFeatureIndex, &winnerCutPoint, &winnerPointsAtLeft,
                      &winnerDataStatistics](Local * v) -> void
    {
        if ((!v->winnerIsLeaf) && (winnerIsLeaf || v->winnerSplitCriterionValue < winnerSplitCriterionValue ||
        (SplitCriterionMath::sFabs(winnerSplitCriterionValue - v->winnerSplitCriterionValue) <= epsilon &&
        winnerFeatureIndex > v->winnerFeatureIndex)))
        {
            winnerIsLeaf = false;
            winnerFeatureIndex = v->winnerFeatureIndex;
            winnerSplitCriterionValue = v->winnerSplitCriterionValue;
            winnerCutPoint = v->winnerCutPoint;
            winnerPointsAtLeft = v->winnerPointsAtLeft;
            winnerDataStatistics = v->winnerDataStatistics;
        }

        delete v;
        v = nullptr;
    } );

    if (winnerIsLeaf || winnerPointsAtLeft < context.minLeafSize || xRowCount - winnerPointsAtLeft < context.minLeafSize)
    {
        makeLeaf(nodeIndex, totalDataStatistics.getBestDependentVariableValue(), context.leavesData.add(totalDataStatistics),
                 static_cast<double>(context.splitCriterion(totalDataStatistics, xRowCount)), static_cast<int>(xRowCount));
        return;
    }

    makeSplit(nodeIndex, winnerFeatureIndex, winnerCutPoint, static_cast<double>(context.splitCriterion(totalDataStatistics, xRowCount)),
              static_cast<int>(xRowCount));

    //left leaf
    makeLeaf(_nodes[nodeIndex].leftChildIndex(), winnerDataStatistics.getBestDependentVariableValue(), context.leavesData.add(winnerDataStatistics),
             static_cast<double>(context.splitCriterion(winnerDataStatistics, winnerPointsAtLeft)), static_cast<int>(winnerPointsAtLeft));

    //right leaf
    totalDataStatistics -= winnerDataStatistics;
    makeLeaf(_nodes[nodeIndex].rightChildIndex(), totalDataStatistics.getBestDependentVariableValue(), context.leavesData.add(totalDataStatistics),
             static_cast<double>(context.splitCriterion(totalDataStatistics, xRowCount - winnerPointsAtLeft)), static_cast<int>(xRowCount - winnerPointsAtLeft));

    if (w) { const_cast<NumericTable *>(w)->releaseBlockOfColumnValues(wBD); }
    const_cast<NumericTable *>(&y)->releaseBlockOfColumnValues(yBD);
    for (size_t i = 0; i < xColumnCount; ++i)
    {
        const_cast<NumericTable *>(&x)->releaseBlockOfColumnValues(xBD[i]);
    }
    delete[] dx;
    delete[] xBD;
    daal_free(indexes);
    dx      = nullptr;
    xBD     = nullptr;
    indexes = nullptr;
    return;
}
