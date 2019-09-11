/* file: InitParameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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


    private native long init(long nClusters, long maxIterations);

    private native long cGetNClusters(long parameterAddress);

    private native long cGetNRowsTotal(long parameterAddress);

    private native long cGetOffset(long parameterAddress);

    private native double cGetOversamplingFactor(long parameterAddress);

    private native long cGetNRounds(long parameterAddress);

    private native void cSetNClusters(long parameterAddress, long nClusters);

    private native void cSetNRowsTotal(long parameterAddress, long nClusters);

    private native void cSetOffset(long parameterAddress, long offset);

    private native void cSetOversamplingFactor(long parameterAddress, double factor);

    private native void cSetNRounds(long parameterAddress, long nRounds);
}
/** @} */
