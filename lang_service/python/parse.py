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

###############################################################################
###############################################################################
# Logic to parse DAAL's C++ headers and extract the information for generating
# a config used for generating SWIG interface files.
#
# The parsing is almost context-free and staggering simple. It is by no means
# intended to parse C++ in any other context.
#
# We define several small parsers, each covering a special purpose (extracting a specific feature).
# Each parser accepts line by line, parses it and potentially fills the globl dict.
# When returning False another parser might be applied, if returning True the
# entire line was consumed and there is nothing more to extract.
# The global dict is the "gdict" attribute of the context argument.
# It maps dict-names to dicts. Each such dict contains the extracted data of interesting objects.
#  - namespace:                                         gdict['ns'])
#  - #include files:                                    gdict['includes'])
#  - classes/structs (even if non-template):            gdict['classes']
#    - public class members:                            gdict['classes'][class].members
#    - get/set methods (formatted for SWIG %rename):    gdict['classes'][class].setgets
#    - template member functions:                       gdict['classes'][class].templates
#  - enum:                                              gdict['enums']
#  - at least one class/func uses a compute method:     gdict['needs_methods']
#  - string of errors encountered (templates)           gdict['error_template_string']
#  - set of steps found (for distributed)               gdict['steps']
# Note that (partial) template class specializations will get separate entires in 'classes'.
# The speicializing template arguments will be appended to the class name and the member
# attribute partial set to true.
#
# Yes: the get/set business needs some reworking.
#
# The context keeps parsing context/state such as the current class. Parser specific
# states should be stored in the parser object itself.
# The context also holds a list of classes/functions which should be ignored.
#
# Expected C++ format/layout
#   - every file defines at most one linear namespace hierachy
#     an error will occur if there is more than one namespace in another
#   - namespace declarations followed by '{' will be ignored
#     - only to be used for forward declarations which will be ignored
#   - the innermost namespace must be 'interface1'
#   - enum values are defined one per separate line
#   - templates are particularly difficult
#     - "template<.*>" should be on a separate line
#     - "template<.*>" must be on a single line
#     - implementation body should start on a separate line
#     - special types we detect:
#       - steps
#       - methods: the argument type must end with 'Method'
#         we do not understand or map the actual type
#   - forward declarations for non-template classes/structs are ignored
#     (such as "class myclass;")
###############################################################################
###############################################################################

from collections import defaultdict
import re

###############################################################################
class cpp_class(object):
    """A C++ class representation"""
    def __init__(self, n, tpl, partial=False):
        self.name = n             # name of the class
        self.template_args = tpl  # list of template arguments as [name, values, default]
        self.members = {}         # dictionary mapping member names to their types
        self.setgets = []         # set and get methods, formatted for %rename
        self.templates = []       # template member functions
        self.partial = partial    # True if this represents a (partial) spezialization
        assert not partial or tpl

###############################################################################


###############################################################################
class ns_parser(object):
    """parse namespace declaration"""
    def parse(self, l, ctxt):
        m = re.match(r'namespace +(\w+)(.*)', l)
        if m and (not m.group(2) or '{' not in m.group(2)):
            ctxt.gdict['ns'].append(m.group(1))
            return True
        return False


class interface_parser(object):
    """Look for 'using' declarations to determine if interface1 or 2 should be used."""
    def parse(self, l, ctxt):
        m = re.match(r'\s*using +(interface\d+)::(\w+);', l)
        if m:
            if m.group(1) != 'interface1':
                ctxt.gdict['namespaces'][m.group(2)] = m.group(1)
            return True
        return False


###############################################################################
class include_parser(object):
    """parse #include"""
    def parse(self, l, ctxt):
        mi = re.match(r'#include\s+[<\"](algorithms/.+?h)[>\"]', l)
        if mi:
            ctxt.gdict['includes'].add(mi.group(1))
            return True
        return False


###############################################################################
class enum_parser(object):
    """Parse an enum"""
    def parse(self, l, ctxt):
        me = re.match(r'\s*enum +(\w+)\s*', l)
        if me:
            ctxt.enum = me.group(1)
            return True
        # if found enum Method, extract the enum values
        if ctxt.enum:
            me = re.match(r'.*?}.*', l)
            if me:
                ctxt.enum = False
                return True
            else:
                me = re.match(r'\s*(\w+)\s*=\s*(\w+).*', l)
                if me:
                    ctxt.gdict['enums'][ctxt.enum][me.group(1)] = me.group(2) if me.group(2) else ''
                    return True
        return False


###############################################################################
class access_parser(object):
    """Parse access specifiers"""
    def parse(self, l, ctxt):
        if ctxt.curr_class:
            am =  re.match(r'\s*(public|private|protected)\s*:\s*', l)
            if am:
                ctxt.access = am.group(1) == 'public'
        return False


###############################################################################
class step_parser(object):
    """Look for distributed steps"""
    def parse(self, l, ctxt):
        m =  re.match(r'.*[<, ](step\d+(Master|Local))[>, ].*', l)
        if m:
            ctxt.gdict['steps'].add(m.group(1))
        return False


###############################################################################
class setget_parser(object):
    """Parse a set/get methods"""
    def parse(self, l, ctxt):
        if ctxt.curr_class and ctxt.access and not ctxt.template:
            mgs = re.match(r'\s*using .+::(get|set);', l)
            if mgs:
                ctxt.gdict['classes'][ctxt.curr_class].setgets.append(l.strip(' ;'))
                #ctxt.gdict['classes'][ctxt.curr_class].setgets.append('// ' + ctxt.curr_class + ': ' + l.strip())
                return True
            mgs = re.match(r'(\s*)([^\(=\s]+\s+)((get|set)(\((\w+).*))', l)
            if mgs:
                ctxt.gdict['classes'][ctxt.curr_class].setgets.append([mgs.group(4), mgs.group(2), mgs.group(6), mgs.group(3)])
                #ctxt.gdict['classes'][ctxt.curr_class].setgets.append('// %rename(' + mgs.group(4) + mgs.group(6).replace('Id', '') + ') /*' + mgs.group(2) + '*/ daal::{{ns}}::interface1::' + ctxt.curr_class + '::' + mgs.group(3) + ';')
                return True
        return False


###############################################################################
class member_parser(object):
    """Parse class members"""
    def parse(self, l, ctxt):
        if ctxt.curr_class and ctxt.access:
            mm = re.match(r'\s*([\w:<>_]+)(?<!return|delete)\s+([\w_]+)\s*;', l)
            if mm :
                if mm.group(2) not in ctxt.gdict['classes'][ctxt.curr_class].members:
                    ctxt.gdict['classes'][ctxt.curr_class].members[mm.group(2)] = mm.group(1)
                return True
        return False


###############################################################################
class global_defines_parser(object):
    """Parse #defined variables from daal_defines.h that should be available globally, including
        DAAL_ALGORITHM_FP_TYPE
        DAAL_DATA_TYPE
        DAAL_SUMMARY_STATISTICS_TYPE
    """
    def parse(self, l, ctxt):

        regex = r'^#define +(\w+) +(\w+) +'
        m = re.search(regex, l)
        if m:
            # Default to double for now
            ctxt.gdict['globals'][m.group(1)] = 'double'
            # ctxt.gdict['globals'][m.group(1)] = m.group(2)


###############################################################################
class class_template_parser(object):
    """Parse a template statement"""
    def parse(self, l, ctxt):
        # not a namespace, no enum Method, let's see if it's a template statement
        # this is checking if we have explicit template instantiation here, which we will simply ignore
        mt = re.match(r'\s*template\s+(class|struct)\s+([\w_]+\s*)+(<[\w_ ,:]+>);', l)
        if mt:
            return True
        # this is checking if we have a template specialization here, which we will simply ignore
        mt = re.match(r'\s*template<>\s*(?!(class|struct))[\w_]+.*', l)
        if mt:
            return True
        # now we test for a "proper" template declaration
        mt = re.match(r'\s*template\s*(<.*?>)', l)
        if mt:
            ctxt.template = mt.group(1)
            # we do not reset ctxt.template unless we mapped it to a class/function
            #    or the next line is nothing we can digest
            # we now do some formatting of common template parameter lists
            tmp = ctxt.template.split(',')
            tmplargs = []
            for ta in tmp:
                tmpltmp = None
                mtm = re.match(r'.*Method +\w+?( *= *(\w+))?[ >]*$', ta)
                if mtm and not 'CompressionMethod' in l:
                    tmpltmp = ['method', 'methods', mtm.group(2) if mtm.group(2) else '']
                    ctxt.gdict['need_methods'] = True
                else:
                    mtt = re.match(r'.*typename \w*?FPType( *= *(\w+))?[ >]*$', ta)
                    if mtt:
                        tmpltmp = ['fptype', 'fptypes', mtt.group(2) if mtt.group(2) else '']
                    else:
                        mtt = re.match(r'.*ComputeStep \w+?( *= *(\w+))?[ >]*$', ta)
                        if mtt:
                            tmpltmp = ['step', 'steps', mtt.group(2) if mtt.group(2) else '']
                if not tmpltmp:
                    tmpltmp = [ta]
                tmplargs.append(tmpltmp)
            ctxt.template = tmplargs
        # we don't have a 'else' (e.g. if not a template) here since we could have template one-liners
        # is it a class/struct?
        m = re.match(r'(?:^\s*|.*?\s+)(class|struct)\s+(DAAL_EXPORT\s+)?(\w+)\s*(<[^>]+>)?(\s*:\s*(public|private|protected).*)?({|$|:|;)', l)
        m2 = re.match(r'\s*(class|struct)\s+\w+;', l) # forward deckarations can be ignored
        if m and not m2:
            if m.group(3) in ctxt.ignores:
                pass
                # error_template_string += fname + ':\n\tignoring ' + m.group(3)
            else:
                # struct/class
                ctxt.curr_class = m.group(3)
                if m.group(4):
                    # template specialization
                    targs = m.group(4).split(',')
                    targs = [a.strip('<> ') for a in targs if not any(ta[0] in a.lower() for ta in ctxt.template)]
                    ctxt.curr_class += '<' + ', '.join(targs) + '>'
                    ctxt.gdict['classes'][ctxt.curr_class] = cpp_class(ctxt.curr_class, ctxt.template, True)
                else:
                    ctxt.gdict['classes'][ctxt.curr_class] = cpp_class(ctxt.curr_class, ctxt.template)
                #elif ctxt.template:
                #        ctxt.gdict['error_template_string'] += '$FNAME:' + str(ctxt.n) + ': Warning: Expected a template specialization for class ' + ctxt.curr_class + '\n'
                if ctxt.template:
                    ctxt.template = False
                ctxt.access = (m.group(1) != 'class')
        elif ctxt.template:
            # we only look for member functions if it's a template
            m = re.match(r'\s*((static|const|inline|DAAL_EXPORT)\s+)*(([:\*&\w<>]| >)+\s+)?[&\*]?(\w+)\s*\(.*', l)
            if m and ctxt.access:
                if m.group(5) not in ctxt.ignores:
                    ctxt.gdict['classes'][ctxt.curr_class].templates.append([ctxt.curr_class + '::' + m.group(5), ctxt.template])
                    ctxt.template = False
                else:
                    pass
                #error_template_string += fname + ':\n\tignoring ' + m.group(5)
            elif ctxt.access and not mt and not m and not any(s in l for s in ctxt.ignores):
                # not a class but a non-mapped template
                ctxt.gdict['error_template_string'] += '$FNAME:' + str(ctxt.n) + ': Warning:\n\t' + str(ctxt.template) + '\n\t' + l
        # the else case means we have a template-statement, class, method to follow next line
        if not mt:
            # let's keep track of occurences of 'template' which we could not digest
            ctxt.template = False
            mt = re.match(r'template[^<]*<', l)
            if mt:
                ctxt.gdict['error_template_string'] += '$FNAME:' + str(ctxt.n) + ': Warning: ' + l
        return False

###############################################################################
class pcontext(object):
    """Parsing context to keep state between lines"""
    def __init__(self, gdict, ignores):
        self.gdict = gdict
        self.ignores = ignores
        self.enum = False
        self.curr_class = False
        self.template = False
        self.access = False


###############################################################################
# parse a DAAL header file and extract information relevant for SWIG interface files
# Common template argument lists are formatted properly for use
#   in interface files config dict
# Also returns string for errors in parsing templates
def parse_header(header, ignores):
    gdict = defaultdict(list)
    gdict.update( { 'ns': [],
                    'namespaces': {},
                    'classes': {},
                    'includes': set(),
                    'steps': set(),
                    'need_methods': False,
                    'error_template_string': '',
                    'enums': defaultdict(lambda: defaultdict(lambda: '')),
                    'globals': {},
                })
    ctxt = pcontext(gdict, ignores)
    parsers = [ns_parser(), include_parser(), enum_parser(), access_parser(), step_parser(), setget_parser(),
               member_parser(), class_template_parser(), global_defines_parser(), interface_parser()]

    # go line by line
    ctxt.n = 1
    for l in header:
        # first strip of eol comments
        l = l.split('//')[0]
        # apply each parser, continue to next line if possible
        for p in parsers:
            if p.parse(l, ctxt):
                break
        ctxt.n += 1

    return gdict
