import numbers
from math import log, exp
from d3m.metadata.hyperparams import Bounded, Enumeration, UniformBool, UniformInt, Uniform, LogUniform, Normal, LogNormal

SAMPLES = 10

def grid(hp):
    """
    Returns a list or range of appropriate HP values, based on the HP type and values.
    Returns None in cases where it is unable to define grid points.
    """
    def is_valid(val):
        try:
            hp.validate(val)
            return True
        except:
            return False

    samples = [ hp.sample() for i in range(0, 100) ]
    sset = set(samples)
    non_numeric = [ x for x in sset if not isinstance(x, numbers.Number) ]
    numeric = [ x for x in sset if isinstance(x, numbers.Number) ]

    # Something went wrong.  We can't influence this HP, so just accept its
    # default behavior.
    if len(sset) == 0:
        return None

    # No numeric values.  Just an enumeration.  Return the values.
    if len(numeric) == 0:
        return non_numeric

    # Numeric, but sample doesn't do anything interesting.  Indeed, this is the
    # intended behavior when the implementer of a primitive derives a HP from
    # an underspecified class, such as Bounded. We'll use some heuristics to
    # nevertheless try to generate some useful values.
    if len(numeric) == 1:
        default_val = numeric[0]
        other_values = [default_val * scale for scale in [0.001, 0.01, 0.1, 0.3, 0.5, 2.0, 1/0.3, 10, 100, 1000]]
        if type(default_val) is int:
            other_values = [int(v) for v in other_values]
        other_values = [v for v in other_values if is_valid(v)]
        return list(set(other_values) | sset)

    # Numeric and we got a range of values.  We'll just sample deciles over 
    # a sufficiently large sample.
    for sample_iter in range(0, 1000):
        if len(numeric) >= 100:
            break
        val = hp.sample()
        if type(val) is int or type(val) is float:
            numeric.append(val)

    n = len(numeric)
    numeric = sorted(numeric)
    numeric = [numeric[int(n/SAMPLES * i)] for i in range(0, SAMPLES)] + [ numeric[-1] ]
    retval = list(set(non_numeric) | set(numeric))
#    print("d3mgrid %s: %s" % (hp, retval))
    return retval
