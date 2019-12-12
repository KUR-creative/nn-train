import sys

def print_if_not(cond, msg): 
    '''
    If cond is not true, then print msg.
    Used for @deal.pre(lambda arg: (...),)
    '''
    if not cond:
        print(msg, file=sys.stderr)

    return cond
