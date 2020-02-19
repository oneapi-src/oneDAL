.. ******************************************************************************
.. * Copyright 2014-2020 Intel Corporation
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

.. daal_rst documentation master file, created by
   sphinx-quickstart on Mon May 20 15:10:10 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |daal-docs| replace:: official Intel\ |reg|\  DAAL website
.. _daal-docs: https://software.intel.com/en-us/intel-daal

.. |github| replace:: |short_name| GitHub\* page
.. _github: https://github.com/intel/daal

|full_name|
------------------------------------------------------

|full_name| (|short_name|) is a library
that helps speed up big data analysis by providing highly optimized
algorithmic building blocks for all stages of data analytics
(preprocessing, transformation, analysis, modeling, validation, and
decision making) in batch, online, and distributed processing modes of
computation. The current version of |short_name| provides
Data Parallel C++ (DPC++) API extensions to the traditional C++ interface.

For general information, visit |github|_. The complete
list of features and documentation are available at |daal-docs|_.

.. toctree::
   :maxdepth: 3
   :caption: Introduction

   getstarted.rst
   build_app/build-application.rst
   legal.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   dev_guide/data-management/data-management.rst
   dev_guide/analysis.rst
   dev_guide/training-prediction.rst
   dev_guide/services/services.rst
   bibliography.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples

   sycl_examples.rst

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/known_issues.rst

.. toctree::
   :maxdepth: 1
   :caption: Contributing Guide

   contribution/coding_guide.rst
