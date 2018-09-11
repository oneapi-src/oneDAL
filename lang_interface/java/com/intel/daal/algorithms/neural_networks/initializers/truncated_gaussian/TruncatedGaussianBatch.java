/* file: TruncatedGaussianBatch.java */
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
 * @defgroup initializers_truncated_gaussian_batch Batch
 * @ingroup initializers_truncated_gaussian
 * @{
 */
/**
 * @brief Contains classes for the truncated gaussian initializer
 */
package com.intel.daal.algorithms.neural_networks.initializers.truncated_gaussian;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.initializers.Input;
import com.intel.daal.algorithms.neural_networks.initializers.Result;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__TRUNCATED_GAUSSIAN__TRUNCATEDGAUSSIANBATCH"></a>
 * \brief Provides methods for truncated gaussian initializer computations in the batch processing mode
 *
 * \par References
 *      - @ref com.intel.daal.algorithms.neural_networks.initializers.Input class
 */
public class TruncatedGaussianBatch extends com.intel.daal.algorithms.neural_networks.initializers.InitializerIface {
    public  TruncatedGaussianParameter    parameter; /*!< Parameters of the truncated gaussian initializer */
    public  TruncatedGaussianMethod       method;    /*!< Computation method for the initializer */
    private Precision    prec;                       /*!< Data type to use in intermediate computations for the initializer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs truncated gaussian initializer by copying input objects and parameters of another truncated gaussian initializer
     * @param context Context to manage the truncated gaussian initializer
     * @param other   An initializer to be used as the source to initialize the input objects
     *                and parameters of this initializer
     */
    public TruncatedGaussianBatch(DaalContext context, TruncatedGaussianBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject));
        parameter = new TruncatedGaussianParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()), prec);
    }

    /**
     * Constructs the truncated gaussian initializer
     * @param context    Context to manage the initializer
     * @param cls        Data type to use in intermediate computations for the initializer, Double.class or Float.class
     * @param method     The initializer computation method, @ref TruncatedGaussianMethod
     * @param mean       The distribution mean
     * @param sigma      The standard deviation of the distribution
     */
    public TruncatedGaussianBatch(DaalContext context, Class<? extends Number> cls, TruncatedGaussianMethod method, double mean, double sigma) {
        super(context);
        construct(context, cls, method, mean, sigma);
    }

    /**
    * Constructs the truncated gaussian initializer
    * @param context    Context to manage the initializer
    * @param cls        Data type to use in intermediate computations for the initializer, Double.class or Float.class
    * @param method     The initializer computation method, @ref TruncatedGaussianMethod
    */
    public TruncatedGaussianBatch(DaalContext context, Class<? extends Number> cls, TruncatedGaussianMethod method) {
        super(context);

        double mean = 0.0;
        double sigma = 1.0;
        construct(context, cls, method, mean, sigma);
    }

    private void construct(DaalContext context, Class<? extends Number> cls, TruncatedGaussianMethod method, double mean, double sigma) {
        this.method = method;

        if (method != TruncatedGaussianMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }
        this.cObject = cInit(prec.getValue(), method.getValue(), mean, sigma);
        input = new Input(context, cGetInput(cObject));
        parameter = new TruncatedGaussianParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()), prec);

        parameter.setMean(mean);
        parameter.setSigma(sigma);
    }

    /**
     * Computes the result of the truncated gaussian initializer
     * @return  TruncatedGaussian initializer result
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated truncated gaussian initializer
     * with a copy of input objects and parameters of this truncated gaussian initializer
     * @param context    Context to manage the initializer
     * @return The newly allocated truncated gaussian initializer
     */
    @Override
    public TruncatedGaussianBatch clone(DaalContext context) {
        return new TruncatedGaussianBatch(context, this);
    }

    private native long cInit(int prec, int method, double mean, double sigma);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
