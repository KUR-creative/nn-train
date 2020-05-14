import funcy as F
import sys

def print_if_not(cond, msg): 
    '''
    If cond is not true, then print msg.
    Used for @deal.pre(lambda arg: (...),)
    '''
    if not cond:
        print(msg, file=sys.stderr)

    return cond

@F.autocurry
def tap(x, f=print):
    '''
    If use your own print function,
    tap(f=my_print)(x)
    '''
    f(x)
    return x
