#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

##  Content:
##     Visual Studio Projects generator for examples
##******************************************************************************

import os
import fnmatch
import uuid
import argparse

_sln_project_decl = '''
Project("{solution_guid}") = "{example_name}", "vcproj\{example_name}\{example_name}.vcxproj", "{example_guid}"
EndProject
'''

_sln_project_platform_decl = '''
        {example_guid}.Debug.dynamic.sequential|x64.ActiveCfg = Debug.dynamic.sequential|x64
        {example_guid}.Debug.dynamic.sequential|x64.Build.0 = Debug.dynamic.sequential|x64
        {example_guid}.Debug.dynamic.threaded|x64.ActiveCfg = Debug.dynamic.threaded|x64
        {example_guid}.Debug.dynamic.threaded|x64.Build.0 = Debug.dynamic.threaded|x64
        {example_guid}.Release.dynamic.sequential|x64.ActiveCfg = Release.dynamic.sequential|x64
        {example_guid}.Release.dynamic.sequential|x64.Build.0 = Release.dynamic.sequential|x64
        {example_guid}.Release.dynamic.threaded|x64.ActiveCfg = Release.dynamic.threaded|x64
        {example_guid}.Release.dynamic.threaded|x64.Build.0 = Release.dynamic.threaded|x64
        {example_guid}.Release.static.sequential|x64.ActiveCfg = Release.static.sequential|x64
        {example_guid}.Release.static.sequential|x64.Build.0 = Release.static.sequential|x64
        {example_guid}.Release.static.threaded|x64.ActiveCfg = Release.static.threaded|x64
        {example_guid}.Release.static.threaded|x64.Build.0 = Release.static.threaded|x64
'''

_this_dir = os.path.dirname(os.path.realpath(__file__))

class cd:
    def __init__(self, new_path):
        self.new_path = os.path.abspath(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)

def get_template(config, template_extension):
    template_base_name = config.template_name
    template_name = os.path.join(config.examples_dir, '{}.{}.tpl'.format(
        template_base_name, template_extension))
    return open(template_name, 'r').read()

def write_file(root_dir, filename, content):
    with open(os.path.join(root_dir, filename), 'w') as f:
        f.write(content)

def get_example_name(relative_example_path):
    example_name, _ = os.path.splitext(os.path.basename(relative_example_path))
    return example_name

def write_proj(config, example_name, proj, proj_filters, proj_user):
    proj_dir = os.path.join(config.output_dir, 'vcproj', example_name)
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)
    print('Write {}.vcxproj'.format(example_name))
    if not config.test:
        write_file(proj_dir, example_name + '.vcxproj', proj)
        write_file(proj_dir, example_name + '.vcxproj.filters', proj_filters)
        write_file(proj_dir, example_name + '.vcxproj.user', proj_user)

def generate_sln(config, examples_info):
    solution_guid = '{' + '{}'.format(uuid.uuid3(uuid.NAMESPACE_URL, config.solution_name)).upper() + '}'
    all_project_decls = []
    all_platform_decl = []
    for name, guid in examples_info:
        project_decl = _sln_project_decl.format(
            solution_guid = solution_guid,
            example_name = name,
            example_guid = guid
        )
        platfrom_decl = _sln_project_platform_decl.format(
            example_guid = guid
        )
        all_project_decls.append(project_decl)
        all_platform_decl.append(platfrom_decl)
    all_project_decls_str = ''.join(all_project_decls)
    all_platform_decl_str = ''.join(all_platform_decl)
    solution = get_template(config, 'sln').format(
        project_decl = all_project_decls_str,
        platform_decl = all_platform_decl_str
    )
    print('Write {}'.format(config.solution_name))
    if not config.test:
        write_file(config.output_dir, config.solution_name, solution)

def generate_proj(config, relative_example_path):
    example_name = get_example_name(relative_example_path)
    normalized_example_path = relative_example_path.replace('/', '\\')
    guid = '{' + '{}'.format(uuid.uuid3(uuid.NAMESPACE_URL, relative_example_path)).upper() + '}'
    proj = get_template(config, 'vcxproj').format(
        example_guid = guid,
        example_name = example_name,
        example_relative_path = normalized_example_path,
    )
    proj_filters = get_template(config, 'vcxproj.filters').format(
        example_relative_path = normalized_example_path,
    )
    proj_user = get_template(config, 'vcxproj.user')
    write_proj(config, example_name, proj, proj_filters, proj_user)
    return example_name, guid

def find_all_examples(examples_dir):
    examples = []
    for root, dirnames, filenames in os.walk(examples_dir):
        for filename in fnmatch.filter(filenames, '*.cpp'):
            rel_path = os.path.relpath(root, examples_dir)
            if filename not in ['jaccard_batch_app.cpp']:
                examples.append(os.path.join(rel_path, filename))
    return examples

def generate(config):
    examples_info = []
    for relative_path in find_all_examples(config.examples_dir):
        guid, name = generate_proj(config, relative_path)
        examples_info.append((guid, name))
    generate_sln(config, examples_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('examples_dir', type=str,
                        help='Path to C++ examples directory')
    parser.add_argument('output_dir', type=str,
                        help='Path to the output directory that will contain generated solution')
    parser.add_argument('solution_name', type=str,
                        help='Solution name to be generated')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Does not write the file, but prints log')
    parser.add_argument('--template_name', type=str, default='daal_win',
                        help='Name of the solution template file without extension')
    config = parser.parse_args()
    generate(config)
