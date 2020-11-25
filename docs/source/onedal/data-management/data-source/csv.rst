.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
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


.. highlight:: cpp
.. default-domain:: cpp

.. _csv-data-source:

---------------
CSV data source
---------------
Class ``csv::data_source`` is an API for accessing the data source represented
as a :term:`csv file <CSV file>`. CSV data source is used with
:expr:`read` operation to extract data in text format from the given input file,
process it using provided parameters (such as delimiter and read options),
transform it into numerical representation, and store it as an in-memory
:txtref:`dataset` of a chosen type.

Supported type of in-memory object for :expr:`read` operation with CSV data
source is :expr:`oneapi::dal::table`.

CSV data source requires input file name to be set in the constructor, while the
other parameters of the constructor such as delimiter and read options rely on
default values.

Usage example
-------------
::

   using namespace oneapi;

   const auto data_source = dal::csv::data_source("data.csv", ',');

   const auto table = dal::read<dal::table>(data_source);


Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::csv`` namespace and be available via inclusion of the
``oneapi/dal/io/csv.hpp`` header file.

::

   enum class read_options : std::uint64_t {
      none = 0,
      parse_header = 1 << 0
   };

   constexpr char default_delimiter = ',';
   constexpr read_options default_read_options = read_options::none;

   class data_source {
   public:
      data_source(const char *file_name,
                  char delimiter = default_delimiter,
                  read_options opts = default_read_options);

      data_source(const std::string &file_name,
                  char delimiter = default_delimiter,
                  read_options opts = default_read_options);

      std::string get_file_name() const;
      char get_delimiter() const;
      read_options get_read_options() const;
   };

.. namespace:: oneapi::dal::csv
.. class:: data_source

   .. function:: data_source(const char *file_name, char delimiter = default_delimiter, read_options opts = default_read_options)

      Creates a new instance of a CSV data source with the given
      :expr:`file_name`, :expr:`delimiter` and read options :expr:`opts` flag.

   .. function:: data_source(const std::string &file_name, char delimiter = default_delimiter, read_options opts = default_read_options)

      Creates a new instance of a CSV data source with the given
      :expr:`file_name`, :expr:`delimiter` and read options :expr:`opts` flag.

   .. member:: std::string file_name = ""

      A string that contains the name of the file with the dataset to read.

      Getter
         | ``std::string get_filename() const``

   .. member:: char delimiter = default_delimiter

      A character that represents the delimiter between separate features in the
      input file.

      Getter
         | ``char get_delimter() const``

   .. member:: read_options options = default_read_options

      Value that stores read options to be applied during reading of the input
      file. Enabled ``parse_header`` option indicates that the first line in the
      input file is processed as a header record with features names.

      Getter
         | ``read_options get_read_options() const``


Reading :expr:`oneapi::dal::read<Object>(...)`
------------------------------------------------

Args
~~~~
::

   template <typename Object>
   class read_args {
   public:
      read_args();
   };

.. namespace:: oneapi::dal::csv
.. class:: template <typename Object> \
           read_args

   .. function:: read_args()

      Creates args for the read operation with the default attribute
      values.

Operation
~~~~~~~~~

:expr:`oneapi::dal::table` is the only supported value of the :code:`Object`
template parameter for :expr:`read` operation with CSV data source.

.. namespace:: oneapi::dal
.. function:: template <typename Object, typename DataSource> \
              Object read(const DataSource& ds)

   :tparam Object: |short_name| object type that is produced as a result of
                   reading from the data source.
   :tparam DataSource: CSV data source :expr:`csv::data_source`.
