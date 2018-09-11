/* file: Parameter.java */
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
public class Parameter extends com.intel.daal.algorithms.Parameter {
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
