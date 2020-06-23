/* file: InitParameter.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * @ingroup kmeans_init
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITPARAMETER"></a>
 * @brief Parameters for computing initial clusters for the K-Means method
 */
public class InitParameter extends com.intel.daal.algorithms.Parameter {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public InitParameter(DaalContext context, long cParameter, long nClusters, long offset) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Constructs a parameter
     * @param context       Context to manage the parameter for computing initial clusters for the K-Means algorithm
     * @param nClusters     Number of clusters
     * @param offset        Offset in the total data set specifying the start of a block stored on a given local node
     */
    public InitParameter(DaalContext context, long nClusters, long offset) {
        super(context);
        initialize(nClusters, offset);
    }

    /**
     * Constructs a parameter
     * @param context       Context to manage the parameter for computing initial clusters for the K-Means algorithm
     * @param nClusters     Number of clusters
     */
    public InitParameter(DaalContext context, long nClusters) {
        super(context);
        initialize(nClusters, 0);
    }

    private void initialize(long nClusters, long offset) {
        this.cObject = init(nClusters, offset);
    }

    /**
     * Retrieves the number of clusters
     * @return Number of clusters
     */
    public long getNClusters() {
        return cGetNClusters(this.cObject);
    }

    /**
     * Retrieves the total number of rows in the distributed processing mode
     * @return Total number of rows
     */
    public long getNRowsTotal() {
        return cGetNRowsTotal(this.cObject);
    }

    /**
     * Retrieves the offset in the total data set specifying the start of a block stored on a given local node
     * @return Offset in the total data set
     */
    public long getOffset() {
        return cGetOffset(this.cObject);
    }

    /**
     * Kmeans|| only. Retrieves fraction of nClusters being chosen in each of nRounds of kmeans||.
     * L = nClusters* oversamplingFactor points are sampled in a round.
     * @return Fraction of nClusters
     */
    public double getOversamplingFactor() {
        return cGetOversamplingFactor(this.cObject);
    }

    /**
     * Kmeans|| only. Retrieves the number of rounds for k-means||.
     * (oversamplingFactor*nRounds) > 1 is a requirement.
     * @return Number of rounds
     */
    public long getNRounds() {
        return cGetNRounds(this.cObject);
    }

    /**
     * Kmeans++ only. The number of trials to generate all clusters but the first initial cluster.
     * @return Number of trials
     */
    public long getNTrials() {
        return cGetNTrials(this.cObject);
    }

    /**
    * Sets the number of clusters
    * @param nClusters Number of clusters
    */
    public void setNClusters(long nClusters) {
        cSetNClusters(this.cObject, nClusters);
    }

    /**
    * Sets the total number of rows in the distributed processing mode
    * @param nRowsTotal Total number of rows
    */
    public void setNRowsTotal(long nRowsTotal) {
        cSetNRowsTotal(this.cObject, nRowsTotal);
    }

    /**
     * Sets the offset in the total data set specifying the start of a block stored on a given local node
     * @param offset Offset in the total data set specifying the start of a block stored on a given local node
     */
    public void setOffset(long offset) {
        cSetOffset(this.cObject, offset);
    }

    /**
     * Kmeans|| only. Sets fraction of nClusters being chosen in each of nRounds of kmeans||.
     * L = nClusters* oversamplingFactor points are sampled in a round.
     * @param factor Fraction of nClusters
     */
    public void setOversamplingFactor(double factor) {
        cSetOversamplingFactor(this.cObject, factor);
    }

    /**
     * Kmeans|| only. Sets the number of rounds for k-means||.
     * (L*nRounds) > 1 is a requirement.
     * @param nRounds Number of rounds
     */
    public void setNRounds(long nRounds) {
        cSetNRounds(this.cObject, nRounds);
    }

    /**
     * Kmeans++ only. The number of trials to generate all clusters but the first initial cluster.
     * @param nTrials Number of trials
     */
    public void setNTrials(long nTrials) {
        cSetNTrials(this.cObject, nTrials);
    }

    private native long init(long nClusters, long maxIterations);

    private native long cGetNClusters(long parameterAddress);

    private native long cGetNRowsTotal(long parameterAddress);

    private native long cGetOffset(long parameterAddress);

    private native double cGetOversamplingFactor(long parameterAddress);

    private native long cGetNTrials(long parameterAddress);

    private native long cGetNRounds(long parameterAddress);

    private native void cSetNClusters(long parameterAddress, long nClusters);

    private native void cSetNRowsTotal(long parameterAddress, long nClusters);

    private native void cSetOffset(long parameterAddress, long offset);

    private native void cSetOversamplingFactor(long parameterAddress, double factor);

    private native void cSetNRounds(long parameterAddress, long nRounds);

    private native void cSetNTrials(long parameterAddress, long nTrials);
}
/** @} */
