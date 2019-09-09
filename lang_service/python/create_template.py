#===============================================================================
# Copyright 2014-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import glob
import os
import re
import subprocess

from os.path import join as jp

template_skeleton_string = """{% from 'jinjadefs.tmpl' import add_getBlocks, doctemplate, add_compute, add_compute2 %}
/* Defines the classes/types/consts etc of the ${package_name_definition} namespace of DAAL
   This is a interface-file for a leaf in the namespace hierachy
    (template and non-template defs together).
   This interface-file is not intended to be included or imported.
*/
%module(package="${package_name}") ${package_tip}

%include daal_common.i

// import the files swig needs (don't include!)
// only import the last module in package hierachy (omit importing its parents)
%import(module="${last_modules}") "${last_modules_tip}.i"

// standard/basic type mappings
%import <std_string.i>

%{
#include <daal.h>
using namespace daal::algorithms;
%}

// Let swig do the heavy-lifting: parse the headers and
//  extract the interfaces
${include_files_string}
${namespace_string_before_brackets}
${commented_include_files_string}
${namespace_string_after_brackets}
"""




if __name__ == "__main__":
    import argparse

    include_directories = ['algorithms', ]
    argParser = argparse.ArgumentParser(prog="swig_tool.py",
                                     description="tool to help automating tasks like creating skeleton templates",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument('-c', '--create_templates', default=False, action='store_true',         help="creates new skeleton template files")
    argParser.add_argument('-n', '--name',             default='algorithms', choices=include_directories, help="name of directory you want to work on")
    args = argParser.parse_args()

    # This will not ever overwrite existing templates, only create non-existing ones
    if args.create_templates:
        # Need to extract namespaces from DAAL header files.
        DAAL_include_root = jp(os.environ['DAALROOT'], 'include')
        chosen_root_dir = jp(DAAL_include_root, args.name) # Header file location, e.g. DAAL/algorithms or DAAL/services
        chosen_root_dir_list = next(os.walk(chosen_root_dir))[1] # List of all directories/namespaces

        for directory in chosen_root_dir_list:
            directory_path = jp(chosen_root_dir, directory)
            header_file_list = os.listdir(directory_path)

            # Make sure the "_type.h" is added first
            for i in range(0, len(header_file_list)):
                if "_types.h" in header_file_list[i]:
                    header_file_list.insert(0, header_file_list.pop(i))
                    break

            root_location = 0
            dicty = {}
            header_file_dict = ()
            for header_file in header_file_list:
                # Trying to find the namespaces in the header file
                grep_namespace = subprocess.check_output(['grep', '-oP', '^namespace \K.*', jp(chosen_root_dir, directory, header_file)])
                grep_namespace = grep_namespace.split('\r')
                namespace_array = [None]*len(grep_namespace)
                for i in range(0, len(grep_namespace)):
                    namespace = re.sub(re.escape('\n'), '', grep_namespace[i])
                    if namespace != '':
                        namespace_array[i] = namespace
                    if namespace == args.name:
                        root_location = i+1
                namespace_array.remove(None)
                header_file_dict += (header_file, namespace_array)

            simple_namespace_flag = True
            for i in range(3, len(header_file_dict), 2):
                if (header_file_dict[1] != header_file_dict[i]):
                    simple_namespace_flag = False
                    break

            if simple_namespace_flag:
                for header_file in header_file_dict:
                    # Finds the names of packages for correct imports
                    package_name = ''
                    package_name_definition = ''
                    package_name += namespace_array[0] + '.'
                    for i in range(1, len(namespace_array)-2):
                        package_name += namespace_array[i] + ('.' if i != (len(namespace_array) - 3) else '')
                        package_name_definition += namespace_array[i] + ('.' if i != (len(namespace_array) - 3) else '')
                    namespace_string_before_brackets = ''
                    for i in range(0, len(namespace_array)):
                        namespace_string_before_brackets += "    " * i
                        namespace_string_before_brackets += "namespace " + str(namespace_array[i]) + " {\n"
                        #for i in range (0, i+1):

                    include_files_string = ''

                    for h in header_file_list:
                        include_files_string += "%include <" + jp(args.name, directory, h) + ">\n"

                    namespace_string_after_brackets = ''
                    for i in range(len(namespace_array), 0, -1):
                        #for i in range(0, i-1):
                        namespace_string_after_brackets += "    " * (i-1)
                        namespace_string_after_brackets += "}\n"

                    commented_include_files_string = ''
                    for h in header_file_list:
                        commented_include_files_string += ("    " * len(namespace_array)) + "// <" + jp(args.name, directory, h) + ">\n"

                    dicty['include_files_string'] = include_files_string
                    dicty['commented_include_files_string'] = commented_include_files_string
                    dicty['package_name'] = package_name
                    dicty['package_name_definition'] = package_name_definition.replace('.', '/') + '/' + namespace_array[-2]
                    dicty['package_tip'] = namespace_array[-2]
                    dicty['last_modules'] = namespace_array[0] + '.' + namespace_array[1] + ('.' + namespace_array[2] if namespace_array[3] != 'interface1' else '')
                    dicty['last_modules_tip'] = namespace_array[2] if namespace_array[3] != 'interface1' else namespace_array[1]
                    dicty['namespace_string_before_brackets'] = namespace_string_before_brackets
                    dicty['namespace_string_after_brackets'] = namespace_string_after_brackets

                    from string import Template
                    t = Template(template_skeleton_string)
                    new_template_skeleton_string = t.substitute(dicty)

                    DAAL_swig_directory = os.path.abspath('.')
                    cur_template_list = glob.glob('*.i.tmpl') # Creates list of template files in current directory (SAT/DAAL/swig)
                    template_file_name = "_".join(namespace_array[root_location:-1])
                    if not os.path.isfile(jp(DAAL_swig_directory, template_file_name + '.i.tmpl')):
                        if "interface1" in template_file_name:
                            with open(jp(DAAL_swig_directory, "errors_file.txt"), "a") as errors_file:
                                errors_file.write(template_file_name)
                                errors_file.write(" should not exist. Must have several namespaces\n")
                        else:
                            print template_file_name + " does not exist. Creating " + jp(os.path.dirname('.'), template_file_name + '.i.tmpl')
                            with open(jp(DAAL_swig_directory, template_file_name + '.i.tmpl'), 'w') as new_template_file:
                                new_template_file.write(str(new_template_skeleton_string))
            if 1 == 4:
                 for header_file in header_file_dict:
                    # Finds the names of packages for correct imports
                    package_name = ''
                    package_name_definition = ''
                    package_name += namespace_array[0] + '.'
                    for i in range(1, len(namespace_array)-2):
                        package_name += namespace_array[i] + ('.' if i != (len(namespace_array) - 3) else '')
                        package_name_definition += namespace_array[i] + ('.' if i != (len(namespace_array) - 3) else '')
                    namespace_string_before_brackets = ''
                    for i in range(0, len(namespace_array)):
                        namespace_string_before_brackets += "namespace " + str(namespace_array[i]) + " {\n"
                        #for i in range (0, i+1):
                        namespace_string_before_brackets += "    " * (i+1)

                    include_files_string = ''
                    if namespace_array[-3] != args.name:
                        include_files_string += "// DELETE UNNECESSARY HEADER FILES\n"
                    for h in header_file_list:
                        include_files_string += "%include <" + jp(args.name, directory, h) + ">\n"

                    namespace_string_after_brackets = ''
                    for i in range(len(namespace_array), 0, -1):
                        #for i in range(0, i-1):
                        namespace_string_after_brackets += "    " * (i-1)
                        namespace_string_after_brackets += "}\n"

                    commented_include_files_string = ''
                    if namespace_array[-3] != args.name:
                        commented_include_files_string += ("    " * len(namespace_array)) + "// DELETE UNNECESSARY HEADER FILES\n"
                    for h in header_file_list:
                        commented_include_files_string += ("    " * len(namespace_array)) + "// <" + jp(args.name, directory, h) + ">\n"

                    dicty['include_files_string'] = include_files_string
                    dicty['commented_include_files_string'] = commented_include_files_string
                    dicty['package_name'] = package_name
                    dicty['package_name_definition'] = package_name_definition.replace('.', '/') + '/' + namespace_array[-2]
                    dicty['package_tip'] = namespace_array[-2]
                    dicty['last_modules'] = namespace_array[0] + '.' + namespace_array[1] + ('.' + namespace_array[2] if namespace_array[3] != 'interface1' else '')
                    dicty['last_modules_tip'] = namespace_array[2] if namespace_array[3] != 'interface1' else namespace_array[1]
                    dicty['namespace_string_before_brackets'] = namespace_string_before_brackets
                    dicty['namespace_string_after_brackets'] = namespace_string_after_brackets

                    from string import Template
                    t = Template(template_skeleton_string)
                    new_template_skeleton_string = t.substitute(dicty)

                    template_file_name = "_".join(namespace_array[root_location:-1])
                    if not os.path.isfile(jp(DAAL_swig_directory, template_file_name + '.i.tmpl')):
                        if "interface1" in template_file_name:
                            with open(jp(DAAL_swig_directory, "errors_file.txt"), "a") as errors_file:
                                errors_file.write(template_file_name)
                                errors_file.write(" should not exist. Must have several namespaces\n")
                        else:
                            print template_file_name + " does not exist. Creating " + jp(os.path.dirname('.'), template_file_name + '.i.tmpl')
                            with open(jp(DAAL_swig_directory, template_file_name + '.i.tmpl'), 'w') as new_template_file:
                                new_template_file.write(str(new_template_skeleton_string))
