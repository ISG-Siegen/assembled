import sys
import traceback
from multiprocessing import Process, Queue


def isolate_function(func, *i_args, **i_kwargs):
    """The Processify Decorate adapted with minor changes to be a callable wrapper

    We also modified it such that it returns the technique args because of random state.

    Taken from and all Credits to https://gist.github.com/schlamar/2311116
    """

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            error = e
            ret = None
        else:
            error = None

        _, _, technique_args, *_ = args
        q.put((ret, error, technique_args))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    q = Queue()
    p = Process(target=process_func, args=tuple([q] + list(i_args)), kwargs=i_kwargs)
    p.start()
    ret, error, technique_args = q.get()
    p.join()

    if error:
        raise RuntimeError("Error in Isolate Subprocess, see previous traceback.")

    return ret, technique_args
