/* file: InitDistributedStep2LocalPlusPlusParameter.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP2LOCALPLUSPLUSPARAMETER"></a>
 * @brief Parameters for computing initial clusters on the step 2 on local nodes. Kmeans++ and || only.
 */
public class InitDistributedStep2LocalPlusPlusParameter extends InitParameter {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public InitDistributedStep2LocalPlusPlusParameter(DaalContext context, long cParameter, long nClusters, boolean bFirstIteration) {
        super(context, nClusters);
        this.cObject = cParameter;
    }

    /**
     * Constructs a parameter
     * @param context           Context to manage the parameter for computing initial clusters for the K-Means algorithm
     * @param nClusters         Number of clusters
     * @param bFirstIteration   True if step2Local is called for the first time
     */
    public InitDistributedStep2LocalPlusPlusParameter(DaalContext context, long nClusters, boolean bFirstIteration) {
        super(context, nClusters);
        initialize(nClusters, bFirstIteration);
    }

    private void initialize(long nClusters, boolean bFirstIteration) {
        this.cObject = init(nClusters, bFirstIteration);
    }

    /**
     * Returns true if step2Local is called for the first time
     * @return Number of clusters
     */
    public boolean getIsFirstIteration() {
        return cGetIsFirstIteration(this.cObject);
    }

    /**
     * Kmeans|| only. Returns true if the last iteration of parallelPlus algorithm processing is performed
     *         and  the output for the 5th step is required
     * @return Total number of rows
     */
    public boolean getOutputForStep5Required() {
        return cGetOutputForStep5Required(this.cObject);
    }

    /**
    * Sets the firstIteration flag
    * @param bFirstIteration firstIteration flag
    */
    public void setIsFirstIteration(boolean bFirstIteration) {
        cSetIsFirstIteration(this.cObject, bFirstIteration);
    }

    /**
    * Sets the outputForStep5Required flag
    * @param bRequired outputForStep5Required flag
    */
    public void setOutputForStep5Required(boolean bRequired) {
        cSetOutputForStep5Required(this.cObject, bRequired);
    }

    private native long init(long nClusters, boolean bFirstIteration);

    private native boolean cGetIsFirstIteration(long parameterAddress);

    private native boolean cGetOutputForStep5Required(long parameterAddress);

    private native void cSetIsFirstIteration(long parameterAddress, boolean bFirstIteration);

    private native void cSetOutputForStep5Required(long parameterAddress, boolean bRequired);
}
/** @} */
