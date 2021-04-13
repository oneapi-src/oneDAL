.. ******************************************************************************
.. * Copyright 2019-2021 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. _glossary:

=========
Glossary
=========

Machine learning terms
======================

.. glossary::
    :sorted:

    Categorical feature
        A :capterm:`feature` with a discrete domain. Can be :capterm:`nominal
        <nominal feature>` or :capterm:`ordinal <ordinal feature>`.

        **Synonyms:** discrete feature, qualitative feature

    Nominal feature
        A :capterm:`categorical feature` without ordering between values. Only
        equality operation is defined for nominal features.

        **Examples:** a person's gender, color of a car

    Classification
        A :capterm:`supervised machine learning problem <supervised learning>`
        of assigning :capterm:`labels <label>` to :capterm:`feature vectors
        <feature vector>`.

        **Examples:** predict what type of object is on the picture (a dog or a cat?),
        predict whether or not an email is spam

    Clustering
        An :capterm:`unsupervised machine learning problem <unsupervised learning>`
        of grouping :capterm:`feature vectors <feature vector>` into bunches, which
        are usually encoded as :capterm:`nominal <nominal feature>` values.

        **Example:** find big star clusters in the space images

    Continuous feature
        A :capterm:`feature` with values in a domain of real numbers. Can be
        :capterm:`interval <interval feature>` or :capterm:`ratio <ratio feature>`

        **Synonyms:** quantitative feature, numerical feature

        **Examples:** a person's height, the price of the house

    Dataset
        A collection of :capterm:`observations <observation>`.

    Dimensionality reduction
        A problem of transforming a set of :capterm:`feature vectors <feature
        vector>` from a high-dimensional space into a low-dimensional space
        while retaining meaningful properties of the original feature vectors.

    Feature
        A particular property or quality of a real object or an event. Has a
        defined type and domain. In machine learning problems, features are
        considered as input variable that are independent from each other.

        **Synonyms:** attribute, variable, input variable

    Feature vector
        A vector that encodes information about real object, an event or a group
        of objects or events. Contains at least one :capterm:`feature`.

        **Example:** A rectangle can be described by two features: its width and
        height

    Inference
        A process of applying a :capterm:`trained <Training>` :capterm:`model`
        to the :capterm:`dataset` in order to predict :capterm:`response` values
        based on input :capterm:`feature vectors <Feature vector>`.

        **Synonym:** prediction

    Inference set
        A :capterm:`dataset` used at the :capterm:`inference` stage.
        Usually without :capterm:`responses <Response>`.

    Interval feature
        A :capterm:`continuous feature` with values that can be compared, added or
        subtracted, but cannot be multiplied or divided.

        **Examples:** a time frame scale, a temperature in Celsius or Fahrenheit

    Label
        A :capterm:`response` with :capterm:`categorical <Categorical feature>` or
        :capterm:`ordinal <Ordinal feature>` values. This is an output in
        :capterm:`classification` and :capterm:`clustering` problems.

        **Example:** the spam-detection problem has a binary label indicating
        whether the email is spam or not

    Model
        An entity that stores information necessary to run :capterm:`inference`
        on a new :capterm:`dataset`. Typically a result of a :capterm:`training`
        process.

        **Example:** in linear regression algorithm, the model contains weight
        values for each input feature and a single bias value

    Observation
        A :capterm:`feature vector` and zero or more :capterm:`responses<Response>`.

        **Synonyms:** instance, sample

    Ordinal feature
        A :capterm:`categorical feature` with defined operations of equality and
        ordering between values.

        **Example:** student's grade

    Outlier
        :capterm:`Observation` which is significantly different from the other
        observations.

    Ratio feature
        A :capterm:`continuous feature` with defined operations of equality,
        comparison, addition, subtraction, multiplication, and division.
        Zero value element means the absence of any value.

        **Example:** the height of a tower

    Regression
        A :capterm:`supervised machine learning problem <Supervised learning>` of
        assigning :capterm:`continuous <Continuous feature>`
        :capterm:`responses<Response>` for :capterm:`feature vectors <Feature vector>`.

        **Example:** predict temperature based on weather conditions

    Response
        A property of some real object or event which dependency from
        :capterm:`feature vector` need to be defined in :capterm:`supervised learning`
        problem. While a :capterm:`feature` is an input in the machine learning
        problem, the response is one of the outputs can be made by the
        :capterm:`model` on the :capterm:`inference` stage.

        **Synonym:** dependent variable

    Supervised learning
        :capterm:`Training` process that uses a :capterm:`dataset` with information
        about dependencies between :capterm:`features <Feature>` and
        :capterm:`responses <Response>`. The goal is to get a :capterm:`model` of
        dependencies between input :capterm:`feature vector` and
        :capterm:`responses <Response>`.

    Training
        A process of creating a :capterm:`model` based on information extracted
        from a :capterm:`training set`. Resulting :capterm:`model` is selected in
        accordance with some quality criteria.

    Training set
        A :capterm:`dataset` used at the :capterm:`training` stage to create a
        :capterm:`model`.

    Unsupervised learning
        :capterm:`Training` process that uses a :capterm:`training set` with no
        :capterm:`responses <Response>`. The goal is to find hidden patters inside
        :capterm:`feature vectors <Feature vector>` and dependencies between them.

    CSV file
        A comma-separated values file (csv) is a type of a text file. Each line in a CSV file is a record containing fields that are separated by the delimiter.
        Fields can be of a numerical or a text format. Text usually refers to categorical values.
        By default, the delimiter is a comma, but, generally, it can be any character.
        For more details, `see <https://en.wikipedia.org/wiki/Comma-separated_values>`_.

|short_name| terms
======================

.. glossary::
    :sorted:

    Accessor
        A |short_name| concept for an object that provides access to the
        data of another object in the special :capterm:`data format`. It abstracts
        data access from interface of an object and provides uniform access to
        the data stored in objects of different types.

    Batch mode
        The computation mode for an algorithm in |short_name|, where all the
        data needed for computation is available at the start and fits the
        memory of the device on which the computations are performed.

    Builder
        A |short_name| concept for an object that encapsulates the creation
        process of another object and enables its iterative creation.

    Contiguous data
        Data that are stored as one contiguous memory block. One of the
        characteristics of a :capterm:`data format`.

    Data format
        Representation of the internal structure of the data.

        **Examples:** data can be stored in array-of-structures or
        compressed-sparse-row format

    Data layout
        A characteristic of :capterm:`data format` which describes the
        order of elements in a :capterm:`contiguous data` block.

        **Example:** row-major format, where elements are stored row by row

    Data type
        An attribute of data used by a compiler to store and access them.
        Includes size in bytes, encoding principles, and available operations
        (in terms of a programming language).

        **Examples:** ``int32_t``, ``float``, ``double``

    Flat data
        A block of :capterm:`contiguous <contiguous data>` :capterm:`homogeneous
        <homogeneous data>` data.

    Getter
        A method that returns the value of the private member variable.

        **Example**:

        .. code-block:: cpp

            std::int64_t get_row_count() const;


    Heterogeneous data
        Data which contain values either of different :capterm:`data types <Data
        type>` or different sets of operations defined on them. One of the
        characteristics of a :capterm:`data format`.

        **Example:** A :capterm:`dataset` with 100
        :capterm:`observations <Observation>` of three :capterm:`interval features <Interval
        feature>`. The first two features are of float32 :capterm:`data type`, while the
        third one is of float64 data type.

    Homogeneous data
        Data with values of single :capterm:`data type` and the same set of
        available operations defined on them. One of the characteristics of a
        :capterm:`data format`.

        **Example:** A :capterm:`dataset` with 100
        :capterm:`observations <Observation>` of three  :capterm:`interval features <Interval
        feature>`, each of type float32

    Immutability
        The object is immutable if it is not possible to change its state after
        creation.

    Metadata
        Information about logical and physical structure of an object. All
        possible combinations of metadata values present the full set of
        possible objects of a given type. Metadata do not expose information
        that is not a part of a type definition, e.g. implementation details.

        **Example:** :capterm:`table` object can contain three :capterm:`nominal features
        <Nominal feature>` with 100 :capterm:`observations <Observation>` (logical
        part of metadata). This object can store data as sparse csr array and
        provides direct access to them (physical part)

    Online mode
        The computation mode for an algorithm in |short_name|, where the
        data needed for computation becomes available in parts over time.

    Reference-counted object
        A copy-constructible and copy-assignable |short_name| object which
        stores the number of references to the unique implementation. Both copy
        operations defined for this object are lightweight, which means that
        each time a new object is created, only the number of references is
        increased. An implementation is automatically freed when the number of
        references becomes equal to zero.

    Setter
        A method that accepts the only parameter and assigns its value to the
        private member variable.

        **Example**:

        .. code-block:: cpp

            void set_row_count(std::int64_t row_count);


    Table
        A |short_name| concept for a :capterm:`dataset` that contains only
        numerical data, :capterm:`categorical <Categorical feature>` or
        :capterm:`continuous <Continuous feature>`. Serves as a transfer of data
        between user's application and computations inside |short_name|.
        Hides details of :capterm:`data format` and generalizes access to the data.

    Workload
        A problem of applying a |short_name| algorithm to a :capterm:`dataset`.

Common oneAPI terms
===================

.. glossary::
    :sorted:

    API
        Application Programming Interface

    DPC++
        Data Parallel C++ (DPC++) is a high-level language designed for data
        parallel programming productivity. DPC++ is based on :term:`SYCL*
        <SYCL>` from the Khronos* Group to support data parallelism and
        heterogeneous programming.

    Host/Device
        OpenCL [OpenCLSpec]_ refers to CPU that controls the connected GPU
        executing kernels.

    JIT
        Just in Time Compilation --- compilation during execution of a program.

    Kernel
        Code written in OpenCL [OpenCLSpec]_ or :term:`SYCL` and executed on a
        GPU device.

    SPIR-V
        Standard Portable Intermediate Representation - V is a language for
        intermediate representation of compute kernels.

    SYCL
        SYCL(TM) [SYCLSpec]_ --- high-level programming model for OpenCL(TM)
        that enables code for heterogeneous processors to be written in a
        "single-source" style using completely standard C++.
