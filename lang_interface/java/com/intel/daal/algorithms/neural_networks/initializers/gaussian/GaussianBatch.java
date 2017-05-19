/* file: GaussianBatch.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * @defgroup initializers_gaussian_batch Batch
 * @ingroup initializers_gaussian
 * @{
 */
/**
 * @brief Contains classes for the gaussian initializer
 */
package com.intel.daal.algorithms.neural_networks.initializers.gaussian;

import com.intel.daal.algorithms.neural_networks.initializers.Input;
import com.intel.daal.algorithms.neural_networks.initializers.Result;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__GAUSSIAN__GAUSSIANBATCH"></a>
 * \brief Provides methods for gaussian initializer computations in the batch processing mode
 *
 * \par References
 *      - @ref com.intel.daal.algorithms.neural_networks.initializers.Input class
 */
public class GaussianBatch extends com.intel.daal.algorithms.neural_networks.initializers.InitializerIface {
    public  GaussianParameter    parameter; /*!< GaussianParameters of the gaussian initializer */
    public  GaussianMethod       method;    /*!< Computation method for the initializer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the initializer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs gaussian initializer by copying input objects and parameters of another gaussian initializer
     * @param context Context to manage the gaussian initializer
     * @param other   An initializer to be used as the source to initialize the input objects
     *                and parameters of this initializer
     */
    public GaussianBatch(DaalContext context, GaussianBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject));
        parameter = new GaussianParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the gaussian initializer
     * @param context    Context to manage the initializer
     * @param cls        Data type to use in intermediate computations for the initializer, Double.class or Float.class
     * @param method     The initializer computation method, @ref GaussianMethod
     * @param a          The distribution mean
     * @param sigma      The standard deviation of the distribution
     * @param seed       The seed for generating random values
     */
    public GaussianBatch(DaalContext context, Class<? extends Number> cls, GaussianMethod method, double a, double sigma, long seed) {
        super(context);
        constructBatch(context, cls, method, a, sigma, seed);
    }

    /**
    * Constructs the gaussian initializer
    * @param context    Context to manage the initializer
    * @param cls        Data type to use in intermediate computations for the initializer, Double.class or Float.class
    * @param method     The initializer computation method, @ref GaussianMethod
    * @param a          The distribution mean
    * @param sigma      The standard deviation of the distribution
    */
    public GaussianBatch(DaalContext context, Class<? extends Number> cls, GaussianMethod method, double a, double sigma) {
        super(context);

        long seed = 777;
        constructBatch(context, cls, method, a, sigma, seed);
    }

    private void constructBatch(DaalContext context, Class<? extends Number> cls, GaussianMethod method, double a, double sigma, long seed) {
        this.method = method;

        if (method != GaussianMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), a, sigma, seed);
        input = new Input(context, cGetInput(cObject));
        parameter = new GaussianParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        parameter.setA(a);
        parameter.setSigma(sigma);
        parameter.setSeed(seed);
    }

    /**
     * Computes the result of the gaussian initializer
     * @return  Gaussian initializer result
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated gaussian initializer
     * with a copy of input objects and parameters of this gaussian initializer
     * @param context    Context to manage the initializer
     * @return The newly allocated gaussian initializer
     */
    @Override
    public GaussianBatch clone(DaalContext context) {
        return new GaussianBatch(context, this);
    }

    private native long cInit(int prec, int method, double a, double sigma, long seed);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
