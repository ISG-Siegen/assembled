import sys
import traceback
from multiprocessing import Process, Queue


def isolate_function(func, *i_args, **i_kwargs):
    """The Processify Decorate adapted with minor changes to be a callable wrapper

    Taken from and all Credits to https://gist.github.com/schlamar/2311116
    """

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception as e:
            error = e
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    q = Queue()
    p = Process(target=process_func, args=tuple([q] + list(i_args)), kwargs=i_kwargs)
    p.start()
    ret, error = q.get()
    p.join()

    if error:
        raise error

    return ret
