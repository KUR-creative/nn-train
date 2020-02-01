from collections import namedtuple
import funcy as F
import itertools as I

tup = lambda f: lambda argtup: f(*argtup)
go = lambda x,*fs: F.rcompose(*fs)(x)
pipe = F.rcompose

def inc(x): return x + 1
def dec(x): return x - 1

def take(n, seq=None):
    return F.take(n,seq) if seq \
    else lambda xs: F.take(n,xs)

def plus(*xs):
    return sum(xs)

def identity(x): return x
def prop(p, obj=None):
    return(getattr(obj, p) if (isinstance(obj,tuple) and 
                               isinstance(p,str))
      else obj[p] if hasattr(obj,'__getitem__')
      else getattr(obj, p) if obj 
      else lambda o: prop(p,o))

def unzip(seq):
    return zip(*seq)

def map(f,*seq):
    return F.map(f,*seq) if seq \
    else lambda *xs: F.map(f,*xs)
def lmap(f,*seq):
    return F.lmap(f,*seq) if seq \
    else lambda *xs: F.lmap(f,*xs)
def tmap(f,*seq):
    return tuple(F.map(f,*seq)) if seq \
    else lambda *xs: tuple(F.map(f,*xs))

def filter(f,*seq):
    return F.filter(f,*seq) if seq \
    else lambda *xs: F.filter(f,*xs)
def lfilter(f,*seq):
    return F.lfilter(f,*seq) if seq \
    else lambda *xs: F.lfilter(f,*xs)
def tfilter(f,*seq):
    return tuple(F.filter(f,*seq)) if seq \
    else lambda *xs: tuple(F.filter(f,*xs))

def remove(f,*seq):
    return F.remove(f,*seq) if seq \
    else lambda *xs: F.remove(f,*xs)
def lremove(f,*seq):
    return F.lremove(f,*seq) if seq \
    else lambda *xs: F.lremove(f,*xs)
def tremove(f,*seq):
    return tuple(F.remove(f,*seq)) if seq \
    else lambda *xs: tuple(F.remove(f,*xs))

def starmap(f,*seq):
    return I.starmap(f,*seq) if seq \
    else lambda *xs: I.starmap(f,*xs)
def lstarmap(f,*seq):
    return list(I.starmap(f,*seq)) if seq \
    else lambda *xs: list(I.starmap(f,*xs))
def tstarmap(f,*seq):
    return tuple(I.starmap(f,*seq)) if seq \
    else lambda *xs: tuple(I.starmap(f,*xs))

def mapcat(f,*seq):
    return F.mapcat(f,*seq) if seq \
    else lambda *xs: F.mapcat(f,*xs)
def lmapcat(f,*seq):
    return F.lmapcat(f,*seq) if seq \
    else lambda *xs: F.lmapcat(f,*xs)
def tmapcat(f,*seq):
    return tuple(F.mapcat(f,*seq)) if seq \
    else lambda *xs: tuple(F.mapcat(f,*xs))

def walk(f,*seq):
    return F.walk(f,*seq) if seq \
    else lambda *xs: F.walk(f,*xs)

def split_with(sep_idxs, li):
    ''' 
    If sep_idxs is empty, then it returns empty generator. 
    But I don't know why..
    '''
    for s,t in F.pairwise( I.chain(sep_idxs, [len(li)]) ):
        yield li[s:t]
def lsplit_with(sep_idxs, li):
    return list(split_with(sep_idxs, li))
def tsplit_with(sep_idxs, li):
    return tuple(split_with(sep_idxs, li))

'''
def partition_with(nums_in_parts, seq):
    ret = []
    iseq = iter(seq)
    for n in nums_in_parts:
        ret.append( take(n,iseq) )
    return ret
'''

def foreach(f,*seq):
    F.lmap(f,*seq)
    return None

def is_empty(coll):
    return (not coll)

def dict2namedtuple(type_name, dic):
    return namedtuple(type_name, sorted(dic))(**dic)

class A():
    def __init__(self,x): self.x = x
    def m(self,x): return x

@F.autocurry
def cut_with_bound(pred, xs):
    chunk = []
    for x in xs:
        chunk.append(x)
        if pred(x):
            yield chunk
            chunk = []
    if chunk: #remaining elements
        yield chunk

assert list(cut_with_bound(lambda x: x == 0, [1,2,0, 3,4,0, 0, 0])) \
    == [[1,2,0], [3,4,0], [0], [0]] 
assert list(cut_with_bound(lambda x: x == 0, [])) \
    == []
assert list(cut_with_bound(lambda x: x == 0, [1,2,0,3,4])) == [[1,2,0],[3,4]], \
    '{} != {}'.format(list(cut_with_bound(lambda x: x == 0, [1,2,0,3,4])), [[1,2,0],[3,4]])
assert list(cut_with_bound(lambda x: x == 0, [1,2,3,4])) == [[1,2,3,4]], \
    '{} != {}'.format(list(cut_with_bound(lambda x: x == 0, [1,2,3,4])), [[1,2,3,4]])
assert list(cut_with_bound(lambda x: x == 0, [0, 1,2,3,4,0])) == [[0], [1,2,3,4,0]], \
    '{} != {}'.format(list(cut_with_bound(lambda x: x == 0, [0, 1,2,3,4,0])), [[0], [1,2,3,4,0]])


if __name__ == '__main__':
    print( lmap(tup(pow))( [(2,5),(3,2),(10,3)] ) )
    print( list(starmap(pow)([(2,5),(3,2),(10,3)] ) ))
    print( lstarmap(pow)([(2,5),(3,2),(10,3)] ) )

    print( list(map(tup(pow))( [(2,5),(3,2),(10,3)] ) ))

    print(F.lmap(
        lambda a,b,c: a+b+c, [1,2,3],[2,3,4],[5,6,7]
    ))

    a = A(2)
    assert 1 == prop(0, {0:1})
    assert '1' == prop(0, {0:'1'})
    assert 2 == prop('x', a)
    assert 0 == prop('m', a)(0)

    assert 1 == prop(0)({0:1})
    assert '1' == prop(0)({0:'1'})
    assert 2 == prop('x')(a)
    assert 0 == prop('m')(a)(0)

    assert 'a' == prop(0,'abcd')
    assert 'a' == prop(0,['a','bcd'])

    from collections import namedtuple
    #namedtuple special rule
    X = namedtuple('X', 'a b c d')
    x = X(1,'b',d='123',c=56.2)
    assert 1     == prop(0,x)
    assert 'b'   == prop('b',x)
    assert '123' == prop('d',x)
    assert 56.2  == prop(2,x), prop(2,x)


    split_with([0,8,16], '0123456789-123456789=1')
    assert list(split_with([0,8,16], '0123456789-123456789=1'))  == ['01234567', '89-12345', '6789=1']
    assert lsplit_with(iter([0,8,16]), '0123456789-123456789=1') == ['01234567', '89-12345', '6789=1']
    assert tsplit_with([0,8,16], '0123456789-123456789=1')       == ('01234567', '89-12345', '6789=1')

    assert tsplit_with([], '0123456789-123456789=1') == ()
    #split_with([], '0123456789-123456789=1')
