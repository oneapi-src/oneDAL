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
 * @ingroup multi_class_classifier
 * @{
 */
/**
 * @brief Contains classes for computing the results of the multi-class classifier
 */
package com.intel.daal.algorithms.multi_class_classifier;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PARAMETER"></a>
 * @brief Parameters of the multi-class classifier algorithm
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

    public Parameter(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets maximum number of iterations of the multi-class classifier training algorithm
     * @param nIter Maximum number of iterations of the multi-class classifier training algorithm
     */
    public void setNIter(long nIter) {
        cSetNIter(this.cObject, nIter);
    }

    /**
     * Retrieves maximum number of iterations of the multi-class classifier training algorithm
     * @return Maximum number of iterations of the multi-class classifier training algorithm
     */
    public long getNIter() {
        return cGetNIter(this.cObject);
    }

    /**
     * Sets convergence threshold of the multi-class classifier training algorithm
     * @param eps Convergence threshold of the multi-class classifier training algorithm
     */
    public void setEps(double eps) {
        cSetEps(this.cObject, eps);
    }

    /**
     * Retrieves convergence threshold of the multi-class classifier training algorithm
     * @return Convergence threshold of the multi-class classifier training algorithm
     */
    public double getEps() {
        return cGetEps(this.cObject);
    }

    /**
     * Sets algorithm for two class classifier model training
     * @param training Algorithm for two class classifier model training
     */
    public void setTraining(com.intel.daal.algorithms.classifier.training.TrainingBatch training) {
        cSetTraining(this.cObject, training.cObject);
    }

    /**
     * Sets algorithm for prediction based on two class classifier model
     * @param prediction Algorithm for prediction based on two class classifier model
     */
    public void setPrediction(com.intel.daal.algorithms.classifier.prediction.PredictionBatch prediction) {
        cSetPrediction(this.cObject, prediction.cObject);
    }

    private native void cSetNIter(long parAddr, long nIter);

    private native long cGetNIter(long parAddr);

    private native void cSetEps(long parAddr, double eps);

    private native double cGetEps(long parAddr);

    private native void cSetTraining(long parAddr, long trainingAddr);

    private native void cSetPrediction(long parAddr, long predictionAddr);

}
/** @} */
