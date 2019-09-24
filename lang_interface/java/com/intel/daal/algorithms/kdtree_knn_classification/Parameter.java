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
 * @ingroup kdtree_knn_classification
 * @{
 */
package com.intel.daal.algorithms.kdtree_knn_classification;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PARAMETER"></a>
 * @brief k nearest neighbors algorithm parameters
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Sets the number of neighbors
     * @param k  Number of neighbors
     */
    public void setK(long k) {
        cSetK(this.cObject, k);
    }

    /**
     * Returns the number of neighbors
     * @return Number of neighbors
     */
    public long getK() {
        return cGetK(this.cObject);
    }

    /**
     * @DAAL_DEPRECATED
     * Sets the seed for random choosing elements from training dataset
     * @param seed   Seed for random choosing elements from training dataset
     */
    public void setSeed(int seed) {
        cSetSeed(this.cObject, seed);
    }

    /**
     * @DAAL_DEPRECATED
     * Returns the seed for random choosing elements from training dataset
     * @return Seed for random choosing elements from training dataset
     */
    public int getSeed() {
        return cGetSeed(this.cObject);
    }

    /**
     * Sets the engine to be used by the algorithm
     * @param engine to be used by the algorithm
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    /**
     * Sets the enable/disable an usage of the input dataset in kNN model flag
     * @param flag   Enable/disable an usage of the input dataset in kNN model flag
     */
    public void setDataUseInModel(DataUseInModelId flag) {
        cSetDataUseInModel(this.cObject, flag.getValue());
    }

    /**
     * Returns the enable/disable an usage of the input dataset in kNN model flag
     * @return Enable/disable an usage of the input dataset in kNN model flag
     */
    public DataUseInModelId getDataUseInModel() {
        return new DataUseInModelId(cGetDataUseInModel(this.cObject));
    }

    private native void cSetK(long algAddr, long k);
    private native void cSetSeed(long algAddr, int seed);
    private native void cSetEngine(long cObject, long cEngineObject);
    private native void cSetDataUseInModel(long algAddr, int flag);

    private native long cGetK(long algAddr);
    private native int cGetSeed(long algAddr);
    private native int cGetDataUseInModel(long algAddr);
}
/** @} */
