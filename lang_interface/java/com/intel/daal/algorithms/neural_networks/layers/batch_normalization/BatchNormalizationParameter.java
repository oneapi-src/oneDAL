/* file: BatchNormalizationParameter.java */
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
 * @ingroup batch_normalization
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONPARAMETER"></a>
 * \brief Class that specifies parameters of the batch normalization layer
 */
public class BatchNormalizationParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     *  Constructs the parameters for the batch normalization layer
     */
    public BatchNormalizationParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public BatchNormalizationParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the smoothing factor of the batch normalization layer
     */
    public double getAlpha() {
        return cGetAlpha(cObject);
    }

    /**
     *  Sets the smoothing factor of the batch normalization layer
     *  @param alpha Smoothing factor that is used in population mean and population variance computations
     */
    public void setAlpha(double alpha) {
        cSetAlpha(cObject, alpha);
    }

    /**
     *  Gets the constant added to the mini-batch variance for numerical stability
     */
    public double getEpsilon() {
        return cGetEpsilon(cObject);
    }

    /**
     *  Sets the constant added to the mini-batch variance for numerical stability
     *  @param epsilon A constant added to the mini-batch variance for numerical stability
     */
    public void setEpsilon(double epsilon) {
       cSetEpsilon(cObject, epsilon);
    }

    /**
     *  Gets the index of the dimension for which the normalization is performed
     */
    public long getDimension() {
        return cGetDimension(cObject);
    }

    /**
     *  Sets the index of the dimension for which the normalization is performed
     *  @param dimension BatchNormalizationIndex of the dimension for which the normalization is performed
     */
    public void setDimension(long dimension) {
       cSetDimension(cObject, dimension);
    }

    /**
     *  Gets the flag that specifies whether the layer is used for the prediction stage or not
     */
    public boolean getPredictionStage() {
        return cGetPredictionStage(cObject);
    }

    /**
     *  Sets the flag that specifies whether the layer is used for the prediction stage or not
     *  @param predictionStage Flag that specifies whether the layer is used for the prediction stage or not
     */
    public void setPredictionStage(boolean predictionStage) {
       cSetPredictionStage(cObject, predictionStage);
    }

    private native long    cInit();
    private native double  cGetAlpha(long cParameter);
    private native void    cSetAlpha(long cParameter, double alpha);
    private native double  cGetEpsilon(long cParameter);
    private native void    cSetEpsilon(long cParameter, double epsilon);
    private native long    cGetDimension(long cParameter);
    private native void    cSetDimension(long cParameter, long dimension);
    private native boolean cGetPredictionStage(long cParameter);
    private native void    cSetPredictionStage(long cParameter, boolean predictionStage);
}
/** @} */
