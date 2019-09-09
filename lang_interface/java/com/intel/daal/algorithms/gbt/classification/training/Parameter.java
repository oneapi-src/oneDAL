/* file: Parameter.java */
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

/**
 * @ingroup gbt_classification_training
 */
/**
 * @brief Contains classes of the gradient boosted trees classification algorithm training
 */
package com.intel.daal.algorithms.gbt.classification.training;

import com.intel.daal.algorithms.gbt.training.SplitMethod;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__TRAINING__PARAMETER"></a>
 * @brief Base class for parameters of the gradient boosted trees classification training algorithm
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Returns split finding method of the gradient boosted trees training algorithm
     * @return Split finding method
     */
    public SplitMethod getSplitMethod() {
        return new SplitMethod(cGetSplitMethod(this.cObject));
    }

    /**
     * Sets split finding method of the gradient boosted trees training algorithm
     * @param splitMethod Split finding method
     */
    public void setSplitMethod(SplitMethod splitMethod) {
        cSetSplitMethod(this.cObject, splitMethod.getValue());
    }

    /**
     * Returns maximal number of iterations parameter of the gradient boosted trees training algorithm
     * @return Number of iterations
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    /**
     * Sets maximal number of iterations for the gradient boosted trees training algorithm
     * @param n Number of iterations
     */
    public void setMaxIterations(long n) {
        cSetMaxIterations(this.cObject, n);
    }

    /**
     * Returns fraction of observations used for a training of one tree, 0 to 1.
     * Default is 1 (sampling without replacement)
     * @return Fraction of observations
     */
    public double getObservationsPerTreeFraction() {
        return cGetObservationsPerTreeFraction(this.cObject);
    }

    /**
     * Sets fraction of observations used for a training of one tree, 0 to 1
     * @param value Fraction of observations
     */
    public void setObservationsPerTreeFraction(double value) {
        cSetObservationsPerTreeFraction(this.cObject, value);
    }

    /**
     * Returns number of features tried as possible splits by the gradient boosted trees classification training algorithm
     * If 0 then sqrt(p) is used, where p is the total number of features.
     * @return Number of features
     */
    public long getFeaturesPerNode() {
        return cGetFeaturesPerNode(this.cObject);
    }

    /**
     * Sets the number of features tried as possible splits by gradient boosted trees classification training algorithm.
     * If 0 then sqrt(p) is used, where p is the total number of features.
     * @param value Number of features
     */
    public void setFeaturesPerNode(long value) {
        cSetFeaturesPerNode(this.cObject, value);
    }

    /**
     * Returns maximal tree depth. Default is 0 (unlimited)
     * @return Maximal tree depth
     */
    public long getMaxTreeDepth() {
        return cGetMaxTreeDepth(this.cObject);
    }

    /**
     * Sets maximal tree depth. Default is 0 (unlimited)
     * @param value Maximal tree depth
     */
    public void setMaxTreeDepth(long value) {
        cSetMaxTreeDepth(this.cObject, value);
    }

    /**
     * Returns minimal number of samples per node. Default is 1
     * @return Minimal number of samples
     */
    public long getMinObservationsInLeafNode() {
        return cGetMinObservationsInLeafNode(this.cObject);
    }

    /**
     * Sets minimal number of samples per node. Default is 1
     * @param value Minimal number of samples
     */
    public void setMinObservationsInLeafNode(long value) {
        cSetMinObservationsInLeafNode(this.cObject, value);
    }

    /**
     * Sets the engine to be used by the algorithm
     * @param engine to be used by the algorithm
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    /**
     * Returns shrinkage parameter (learning rate) of the training procedure
     * Scales the contribution of each tree by a factor (0, 1].
     * Default is 0.3
     * @return Shrinkage value
     */
    public double getShrinkage() {
        return cGetShrinkage(this.cObject);
    }

    /**
     * Sets shrinkage parameter value
     * @param value Shrinkage value
     */
    public void setShrinkage(double value) {
        cSetShrinkage(this.cObject, value);
    }

    /**
     * Returns loss regularization parameter of the training procedure.
     * Min loss reduction required to make a further partition on a leaf node of the tree.
     * Range: [0, inf). Default is 0
     * @return minSplitLoss value
     */
    public double getMinSplitLoss() {
        return cGetMinSplitLoss(this.cObject);
    }

    /**
     * Sets minSplitLoss parameter value
     * @param value minSplitLoss value
     */
    public void setMinSplitLoss(double value) {
        cSetMinSplitLoss(this.cObject, value);
    }

    /**
     * Returns lambda parameter of the training procedure (L2 regularization on weights lambda)
     * Range: [0, inf). Default is 1
     * @return lambda value
     */
    public double getLambda() {
        return cGetLambda(this.cObject);
    }

    /**
     * Sets lambda parameter value
     * @param value lambda value
     */
    public void setLambda(double value) {
        cSetLambda(this.cObject, value);
    }

    /**
     * Returns maximal number of discrete bins to bucket continuous features.
     * Used with 'inexact' split finding method only.
     * Default is 256. Increasing the number results in higher computation costs
     * @return Maximal number of discrete bins to bucket continuous features
     */
    public long getMaxBins()
    {
        return cGetMaxBins(this.cObject);
    }

    /**
     * Sets maximal number of discrete bins to bucket continuous features.
     * Used with 'inexact' split finding method only.
     * Default is 256. Increasing the number results in higher computation costs
     * @param value Maximal number of discrete bins to bucket continuous features
     */
    public void setMaxBins(long value)
    {
        cSetMaxBins(this.cObject, value);
    }

    /**
     * Returns minimal number of observations in a bin. Default is 5
     * Used with 'inexact' split finding method only.
     * @return Minimal number of observations in a bin
     */
    public long getMinBinSize()
    {
        return cGetMinBinSize(this.cObject);
    }

    /**
     * Sets minimal number of observations in a bin. Default is 5
     * Used with 'inexact' split finding method only.
     * @param value Minimal number of observations in a bin
     */
    public void setMinBinSize(long value)
    {
        cSetMinBinSize(this.cObject, value);
    }

    private native int  cGetSplitMethod(long parAddr);
    private native void cSetSplitMethod(long parAddr, int value);

    private native long cGetMaxIterations(long parAddr);
    private native void cSetMaxIterations(long parAddr, long value);

    private native void   cSetObservationsPerTreeFraction(long parAddr, double value);
    private native double cGetObservationsPerTreeFraction(long parAddr);

    private native long cGetFeaturesPerNode(long parAddr);
    private native void cSetFeaturesPerNode(long parAddr, long value);

    private native long cGetMaxTreeDepth(long parAddr);
    private native void cSetMaxTreeDepth(long parAddr, long value);

    private native long cGetMinObservationsInLeafNode(long parAddr);
    private native void cSetMinObservationsInLeafNode(long parAddr, long value);

    private native void cSetEngine(long cObject, long cEngineObject);

    private native void   cSetShrinkage(long parAddr, double value);
    private native double cGetShrinkage(long parAddr);

    private native void   cSetMinSplitLoss(long parAddr, double value);
    private native double cGetMinSplitLoss(long parAddr);

    private native void   cSetLambda(long parAddr, double value);
    private native double cGetLambda(long parAddr);

    private native long cGetMaxBins(long parAddr);
    private native void cSetMaxBins(long parAddr, long value);

    private native long cGetMinBinSize(long parAddr);
    private native void cSetMinBinSize(long parAddr, long value);
}
/** @} */
