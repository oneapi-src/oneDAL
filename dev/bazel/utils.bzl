
def set(iterable):
    return {element: None for element in iterable}

def unique(iterable):
    """Remove duplicates from a list."""
    unique_elements = set(iterable)
    return unique_elements.keys()

def add_prefix(prefix, lst):
    return [ prefix + str(x) for x in lst ]

def warn(msg):
    """Output warning."""
    yellow = "\033[1;33m"
    no_color = "\033[0m"
    print("\n%sWARNING:%s %s\n" % (yellow, no_color, msg))

utils = struct(
    set = set,
    unique = unique,
    add_prefix = add_prefix,
    warn = warn,
)
