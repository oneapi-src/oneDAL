/* file: Parameter.java */
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
 * @ingroup kmeans_compute
 * @{
 */
package com.intel.daal.algorithms.kmeans;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__PARAMETER"></a>
 * @brief Parameters of the K-Means computation method
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    DistanceType distanceType;

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
     * @param context               Context to manage the parameter of the K-Means algorithm
     * @param nClusters             Number of clusters
     * @param maxIterations         Number of iterations
     * @param accuracyThreshold     Threshold for the termination of the algorithm
     * @param gamma                 Weight used in distance calculation for categorical features
     * @param distanceType          Distance used in the algorithm
     * @param resultsToEvaluate     64 bit integer flag that indicates the results to compute
     */
    public Parameter(DaalContext context, long nClusters, long maxIterations, double accuracyThreshold, double gamma,
            DistanceType distanceType, long resultsToEvaluate) {
        super(context);
        this.distanceType = distanceType;
        boolean assignFlag = true;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag, resultsToEvaluate);
    }


    /**
     * Constructs a parameter
     * @param context               Context to manage the parameter of the K-Means algorithm
     * @param nClusters             Number of clusters
     * @param maxIterations         Number of iterations
     * @param accuracyThreshold     Threshold for the termination of the algorithm
     * @param gamma                 Weight used in distance calculation for categorical features
     * @param distanceType          Distance used in the algorithm
     * @param assignFlag            Flag to enable assignment of observations to clusters; assigns data points
     */
    public Parameter(DaalContext context, long nClusters, long maxIterations, double accuracyThreshold, double gamma,
            DistanceType distanceType, boolean assignFlag) {
        super(context);
        this.distanceType = distanceType;
        long resultsToEvaluate = ResultsToComputeId.computeCentroids | ResultsToComputeId.computeAssignments;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag, resultsToEvaluate);
    }

    /**
     * Constructs a parameter
     * @param context               Context to manage the parameter of the K-Means algorithm
     * @param nClusters             Number of clusters
     * @param maxIterations         Number of iterations
     * @param accuracyThreshold     Threshold for the termination of the algorithm
     * @param gamma                 Weight used in distance calculation for categorical features
     * @param distanceType          Distance used in the algorithm
     */
    public Parameter(DaalContext context, long nClusters, long maxIterations, double accuracyThreshold, double gamma,
            DistanceType distanceType) {
        super(context);
        this.distanceType = distanceType;

        boolean assignFlag = true;
        long resultsToEvaluate = ResultsToComputeId.computeCentroids + ResultsToComputeId.computeAssignments;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag, resultsToEvaluate);
    }

    /**
    * Constructs a parameter
    * @param context               Context to manage the parameter of the K-Means algorithm
    * @param nClusters             Number of clusters
    * @param maxIterations         Number of iterations
    * @param accuracyThreshold     Threshold for the termination of the algorithm
    * @param gamma                 Weight used in distance calculation for categorical features
    */
    public Parameter(DaalContext context, long nClusters, long maxIterations, double accuracyThreshold, double gamma) {
        super(context);
        boolean assignFlag = true;
        long resultsToEvaluate = ResultsToComputeId.computeCentroids | ResultsToComputeId.computeAssignments;
        this.distanceType = DistanceType.euclidean;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag, resultsToEvaluate);
    }

    /**
    * Constructs a parameter
    * @param context               Context to manage the parameter of the K-Means algorithm
    * @param nClusters             Number of clusters
    * @param maxIterations         Number of iterations
    * @param accuracyThreshold     Threshold for the termination of the algorithm
    */
    public Parameter(DaalContext context, long nClusters, long maxIterations, double accuracyThreshold) {
        super(context);

        boolean assignFlag = true;
        long resultsToEvaluate = ResultsToComputeId.computeCentroids | ResultsToComputeId.computeAssignments;
        this.distanceType = DistanceType.euclidean;
        double gamma = 1.0;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag, resultsToEvaluate);
    }

    /**
    * Constructs a parameter
    * @param context               Context to manage the parameter of the K-Means algorithm
    * @param nClusters             Number of clusters
    * @param maxIterations         Number of iterations
    */
    public Parameter(DaalContext context, long nClusters, long maxIterations) {
        super(context);
        DistanceType distanceType = DistanceType.euclidean;
        this.distanceType = distanceType;

        boolean assignFlag = true;
        long resultsToEvaluate = ResultsToComputeId.computeCentroids | ResultsToComputeId.computeAssignments;
        double gamma = 1.0;
        double accuracyThreshold = 0.0;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag, resultsToEvaluate);
    }

    private void initialize(long nClusters, long maxIterations, double accuracyThreshold, double gamma,
            boolean assignFlag, long resultsToEvaluate) {
        if (this.distanceType == DistanceType.euclidean) {
            this.cObject = initEuclidean(nClusters, maxIterations);
        } else {
            throw new IllegalArgumentException("distanceType unsupported");
        }

        setAccuracyThreshold(accuracyThreshold);
        setGamma(gamma);
        setAssignFlag(assignFlag);
        setResultsToEvaluate(resultsToEvaluate);
    }

    /**
     * Returns the distance type
     * @return Distance type
     */
    public DistanceType getDistanceType() {
        return distanceType;
    }

    /**
     * Retrieves the number of clusters
     * @return Number of clusters
     */
    public long getNClusters() {
        return cGetNClusters(this.cObject);
    }

    /**
     * Retrieves the number of iterations
     * @return Number of iterations
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    /**
     * Retrieves the threshold for the termination of the algorithm
     * @return Threshold for the termination of the algorithm
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Retrieves the weight used in distance calculation for categorical features
     * @return Weight used in distance calculation for categorical features
     */
    public double getGamma() {
        return cGetGamma(this.cObject);
    }

    /**
     * Retrieves the flag for the assignment of data points
     * @return Flag for the assignment of data points
     */
    public boolean getAssignFlag() {
        return cGetAssignFlag(this.cObject);
    }

    /**
     * Retrieves the 64 bit integer flag that indicates the results to compute
     * @return The 64 bit integer flag that indicates the results to compute
     */
    public long getResultsToEvaluate() {
        return cGetResultsToEvaluate(this.cObject);
    }

    /**
    * Sets the number of clusters
    * @param nClusters Number of clusters
    */
    public void setNClusters(long nClusters) {
        cSetNClusters(this.cObject, nClusters);
    }

    /**
     * Sets the number of iterations
     * @param max Number of iterations.
     */
    public void setMaxIterations(long max) {
        cSetMaxIterations(this.cObject, max);
    }

    /**
     * Sets the threshold for the termination of the algorithm
     * @param accuracy Threshold for the termination of the algorithm
     */
    public void setAccuracyThreshold(double accuracy) {
        cSetAccuracyThreshold(this.cObject, accuracy);
    }

    /**
     * Sets the weight used in distance calculation for categorical features
     * @param gamma Weight used in distance calculation for categorical features
     */
    public void setGamma(double gamma) {
        cSetGamma(this.cObject, gamma);
    }

    /**
     * Sets the flag for the assignment of data points
     * @param assignFlag Flag to enable assignment of observations to clusters
     */
    public void setAssignFlag(boolean assignFlag) {
        cSetAssignFlag(this.cObject, assignFlag);
    }

    /**
     * Sets the 64 bit integer flag that indicates the results to compute
     * @param setResultsToEvaluate The 64 bit integer flag that indicates the results to compute
     */
    public void setResultsToEvaluate(long setResultsToEvaluate) {
        cSetResultsToEvaluate(this.cObject, setResultsToEvaluate);
    }

    private native long initEuclidean(long nClusters, long maxIterations);

    private native long cGetNClusters(long parameterAddress);

    private native long cGetMaxIterations(long parameterAddress);

    private native double cGetAccuracyThreshold(long parameterAddress);

    private native double cGetGamma(long parameterAddress);

    private native boolean cGetAssignFlag(long parameterAddress);

    private native long cGetResultsToEvaluate(long parameterAddress);

    private native void cSetNClusters(long parameterAddress, long nClusters);

    private native void cSetMaxIterations(long parameterAddress, long maxIterations);

    private native void cSetAccuracyThreshold(long parameterAddress, double accuracyThreshold);

    private native void cSetGamma(long parameterAddress, double gamma);

    private native void cSetAssignFlag(long parameterAddress, boolean assignFlag);

    private native void cSetResultsToEvaluate(long parameterAddress, long resultsToCompute);
}
/** @} */
