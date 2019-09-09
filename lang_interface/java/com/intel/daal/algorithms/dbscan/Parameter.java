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
 * @ingroup dbscan_compute
 * @{
 */
package com.intel.daal.algorithms.dbscan;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__PARAMETER"></a>
 * @brief Parameters of the DBSCAN computation method
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Constructs a parameter
     * @param context               Context to manage the parameter of the DBSCAN algorithm
     */
    public Parameter(DaalContext context) {
        super(context);

        double epsilon = 0.0;
        long minObservations = 0;
        boolean memorySavingMode = false;
        long resultsToCompute = 0;
        long blockIndex = 0;
        long nBlocks = 0;
        long leftBlocks = 0;
        long rightBlocks = 0;
        initialize(epsilon, minObservations, memorySavingMode, resultsToCompute, blockIndex, nBlocks, leftBlocks, rightBlocks);
    }

    /**
     * Constructs a parameter
     * @param context               Context to manage the parameter of the DBSCAN algorithm
     * @param epsilon               Radius of neighborhood
     * @param minObservations       Minimal number of observations in neighborhood of core observation
     */
    public Parameter(DaalContext context, double epsilon, long minObservations) {
        super(context);

        boolean memorySavingMode = false;
        long resultsToCompute = 0;
        long blockIndex = 0;
        long nBlocks = 0;
        long leftBlocks = 0;
        long rightBlocks = 0;
        initialize(epsilon, minObservations, memorySavingMode, resultsToCompute, blockIndex, nBlocks, leftBlocks, rightBlocks);
    }

    /**
     * Constructs a parameter
     * @param context               Context to manage the parameter of the DBSCAN algorithm
     * @param epsilon               Radius of neighborhood
     * @param minObservations       Minimal number of observations in neighborhood of core observation
     * @param memorySavingMode      If true then use memory saving (but slower) mode
     */
    public Parameter(DaalContext context, double epsilon, long minObservations, boolean memorySavingMode) {
        super(context);

        long resultsToCompute = 0;
        long blockIndex = 0;
        long nBlocks = 0;
        long leftBlocks = 0;
        long rightBlocks = 0;
        initialize(epsilon, minObservations, memorySavingMode, resultsToCompute, blockIndex, nBlocks, leftBlocks, rightBlocks);
    }

    private void initialize(double epsilon, long minObservations, boolean memorySavingMode,
                            long resultsToCompute, long blockIndex, long nBlocks, long leftBlocks, long rightBlocks) {
        setEpsilon(epsilon);
        setMinObservations(minObservations);
        setMemorySavingMode(memorySavingMode);
        setResultsToCompute(resultsToCompute);
        setBlockIndex(blockIndex);
        setNBlocks(nBlocks);
        setLeftBlocks(leftBlocks);
        setRightBlocks(rightBlocks);
    }

    /**
     * Retrieves the radius of neighborhood
     * @return Radius of neighborhood
     */
    public double getEpsilon() {
        return cGetEpsilon(this.cObject);
    }

    /**
     * Retrieves the minimal number of observations in neighborhood of core observation
     * @return Minimal number of observations in neighborhood of core observation
     */
    public long getMinObservations() {
        return cGetMinObservations(this.cObject);
    }

    /**
     * Retrieves the memory saving mode flag value
     * @return Flag for the memory saving mode
     */
    public boolean getMemorySavingMode() {
        return cGetMemorySavingMode(this.cObject);
    }

    /**
     * Retrieves the 64 bit integer flag that indicates the results to compute
     * @return The 64 bit integer flag that indicates the results to compute
     */
    public long getResultsToCompute() {
        return cGetResultsToCompute(this.cObject);
    }

    /**
     * Retrieves the unique identifier of block initially passed for computation on the local node
     * @return Unique identifier of block initially passed for computation on the local node
     */
    public long getBlockIndex() {
        return cGetBlockIndex(this.cObject);
    }

    /**
     * Retrieves the number of blocks initially passed for computation on all nodes
     * @return Number of blocks initially passed for computation on all nodes
     */
    public long getNBlocks() {
        return cGetNBlocks(this.cObject);
    }

    /**
     * Retrieves the number of blocks that will process observations with value of selected
     * split feature lesser than selected split value
     * @return Number of blocks that will process observations with value of selected
     *         split feature lesser than selected split value
     */
    public long getLeftBlocks() {
        return cGetLeftBlocks(this.cObject);
    }

    /**
     * Retrieves the number of blocks that will process observations with value of selected
     * split feature greater than selected split value
     * @return Number of blocks that will process observations with value of selected
     *         split feature greater than selected split value
     */
    public long getRightBlocks() {
        return cGetRightBlocks(this.cObject);
    }

    /**
    * Sets the radius of neighborhood
    * @param epsilon Radius of neighborhood
    */
    public void setEpsilon(double epsilon) {
        cSetEpsilon(this.cObject, epsilon);
    }

    /**
     * Sets the minimal number of observations in neighborhood of core observation
     * @param minObservations Minimal number of observations in neighborhood of core observation
     */
    public void setMinObservations(long minObservations) {
        cSetMinObservations(this.cObject, minObservations);
    }

    /**
     * Sets the memory saving mode flag
     * @param memorySavingMode Flag for the memory saving mode
     */
    public void setMemorySavingMode(boolean memorySavingMode) {
        cSetMemorySavingMode(this.cObject, memorySavingMode);
    }

    /**
     * Sets the 64 bit integer flag that indicates the results to compute
     * @param resultsToCompute The 64 bit integer flag that indicates the results to compute
     */
    public void setResultsToCompute(long resultsToCompute) {
        cSetResultsToCompute(this.cObject, resultsToCompute);
    }

    /**
     * Sets the unique identifier of block initially passed for computation on the local node
     * @param blockIndex Unique identifier of block initially passed for computation on the local node
     */
    public void setBlockIndex(long blockIndex) {
        cSetBlockIndex(this.cObject, blockIndex);
    }

    /**
     * Sets the number of blocks initially passed for computation on all nodes
     * @param nBlocks Number of blocks initially passed for computation on all nodes
     */
    public void setNBlocks(long nBlocks) {
        cSetNBlocks(this.cObject, nBlocks);
    }

    /**
     * Sets the number of blocks that will process observations with value of selected
     * split feature lesser than selected split value
     * @param leftBlocks Number of blocks that will process observations with value of selected
     *                   split feature lesser than selected split value
     */
    public void setLeftBlocks(long leftBlocks) {
        cSetLeftBlocks(this.cObject, leftBlocks);
    }

    /**
     * Sets the number of blocks that will process observations with value of selected
     * split feature greater than selected split value
     * @param rightBlocks Number of blocks that will process observations with value of selected
     *                    split feature greater than selected split value
     */
    public void setRightBlocks(long rightBlocks) {
        cSetRightBlocks(this.cObject, rightBlocks);
    }

    private native double cGetEpsilon(long parameterAddress);
    private native long cGetMinObservations(long parameterAddress);
    private native boolean cGetMemorySavingMode(long parameterAddress);
    private native long cGetResultsToCompute(long parameterAddress);
    private native long cGetBlockIndex(long parameterAddress);
    private native long cGetNBlocks(long parameterAddress);
    private native long cGetLeftBlocks(long parameterAddress);
    private native long cGetRightBlocks(long parameterAddress);

    private native void cSetEpsilon(long parameterAddress, double epsilon);
    private native void cSetMinObservations(long parameterAddress, long minObservations);
    private native void cSetMemorySavingMode(long parameterAddress, boolean memorySavingMode);
    private native void cSetResultsToCompute(long parameterAddress, long resultsToCompute);
    private native void cSetBlockIndex(long parameterAddress, long blockIndex);
    private native void cSetNBlocks(long parameterAddress, long nBlocks);
    private native void cSetLeftBlocks(long parameterAddress, long leftBlocks);
    private native void cSetRightBlocks(long parameterAddress, long rightBlocks);
}
/** @} */
