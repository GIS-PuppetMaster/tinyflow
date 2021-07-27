from __future__ import absolute_import

from ._base import _LIB, check_call, c_array
import ctypes
import numpy as np
import pynvml
import os


class DLContext(ctypes.Structure):
    """DL context strucure."""
    _fields_ = [("device_id", ctypes.c_int),
                ("device_type", ctypes.c_int)]

    MASK2STR = {
        1: 'cpu',
        2: 'gpu',
    }

    def __init__(self, device_id, device_type):
        super(DLContext, self).__init__()
        self.device_id = device_id
        self.device_type = device_type

    def __repr__(self):
        return "%s(%d)" % (
            DLContext.MASK2STR[self.device_type], self.device_id)


class DLArray(ctypes.Structure):
    """DLArray in C API"""
    _fields_ = [("data", ctypes.c_void_p),
                ("ctx", DLContext),
                ("ndim", ctypes.c_int),
                ("shape", ctypes.POINTER(ctypes.c_int64))]


DLArrayHandle = ctypes.POINTER(DLArray)


def cpu(dev_id=0):
    """Construct a CPU device
    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return DLContext(dev_id, 1)


def gpu(dev_id=0):
    """Construct a CPU device
    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return DLContext(dev_id, 2)


def is_gpu_ctx(ctx):
    """Return if context is GPU context.
    Parameters
    ----------
    ctx : DLContext
        The query context
    """
    return ctx and ctx.device_type == 2


class NDArray(object):
    """Lightweight NDArray class of DL runtime.
    Strictly this is only an Array Container(a buffer object)
    No arthimetic operations are defined.
    """
    __slots__ = ["handle","is_freed"]

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle
        Parameters
        ----------
        handle : DLArrayHandle
            the handle to the underlying C++ DLArray
        """
        self.handle = handle
        self.is_freed = False

    def __del__(self):
        if self.is_freed:
            return
        check_call(_LIB.DLArrayFree(self.handle))


    def free_gpu(self):
        self.is_freed = True
        check_call(_LIB.DLArrayFree(self.handle))


    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i]
                     for i in range(self.handle.contents.ndim))

    @property
    def ctx(self):
        """context of this array"""
        return self.handle.contents.ctx

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArray):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self._sync_copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def _sync_copyfrom(self, source_array):
        """Peform an synchronize copy from the array.
        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=np.float32)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported'
                                % str(type(source_array)))
        source_array = np.ascontiguousarray(source_array, dtype=np.float32)
        if source_array.shape != self.shape:
            raise ValueError('array shape do not match the shape of NDArray')
        source_arr, shape = NDArray._numpyasarray(source_array)
        check_call(_LIB.DLArrayCopyFromTo(
            ctypes.byref(source_arr), self.handle, None))
        # de-allocate shape until now
        _ = shape

    @staticmethod
    def _numpyasarray(np_data):
        """Return a DLArray representation of a numpy array."""
        data = np_data
        assert data.flags['C_CONTIGUOUS']
        arr = DLArray()
        shape = c_array(ctypes.c_int64, data.shape)
        arr.data = data.ctypes.data_as(ctypes.c_void_p)
        arr.shape = shape
        arr.ndim = data.ndim
        # CPU device
        arr.ctx = cpu(0)
        return arr, shape

    def asnumpy(self):
        """Convert this array to numpy array
        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        np_arr = np.empty(self.shape, dtype=np.float32)
        arr, shape = NDArray._numpyasarray(np_arr)
        check_call(_LIB.DLArrayCopyFromTo(
            self.handle, ctypes.byref(arr), None))
        _ = shape
        return np_arr
    #如果target是DLContext，用isinstance==int判断
    #如果target是NDArray,不用判断
    def copyto(self, target, cudaStream):
        """Copy array to target
        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.
        """
        if isinstance(target, DLContext):
            target = empty(self.shape, target)
            if isinstance(target, int):
                return target
        if isinstance(target, NDArray):

            check_call(_LIB.DLArrayCopyFromTo(
                self.handle, target.handle, cudaStream))

        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target

    def old_copyto(self, target):
        """Copy array to target
        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.
        """
        if isinstance(target, DLContext):
            target = empty(self.shape, target)
            if isinstance(target, int):
                return target
        if isinstance(target, NDArray):

            check_call(_LIB.DLArrayCopyFromTo(
                self.handle, target.handle, None))

        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target




#用isinstance==int判断是否超内存
def array(arr, ctx=cpu(0),maxmem=-1):
    """Create an array from source arr.
    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from
    ctx : DLContext, optional
        The device context to create the array
    Returns
    -------
    ret : NDArray
        The created array
    """
    # if maxmem>0:
    #     pynvml.nvmlInit()
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(0)   #GPU0上面跑
    #     info_list = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    #     gpu_memory_used = 0
    #     for info_i in info_list:
    #         if info_i.pid == os.getpid():  # 如果与需要记录的pid一致
    #             gpu_memory_used += info_i.usedGpuMemory
    #     pynvml.nvmlShutdown()  # 最后关闭管理工具
    #     if gpu_memory_used > maxmem:
    #         return int(gpu_memory_used - maxmem)



    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    ret = empty(arr.shape, ctx)
    #
    if isinstance(ret,int):
        return ret

    ret._sync_copyfrom(arr)
    return ret

#用isinstance==int判断是否超内存
def empty(shape, ctx=cpu(0),maxmem=-1):
    """Create an empty array given shape and device
    Parameters
    ----------
    shape : tuple of int
        The shape of the array
    ctx : DLContext
        The context of the array
    Returns
    -------
    arr : ndarray
        The array dlsys supported.
    申请失败，返回size
    """
    if maxmem > 0:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info_list = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        gpu_memory_used = 0
        for info_i in info_list:
            if info_i.pid == os.getpid():  # 如果与需要记录的pid一致
                gpu_memory_used += info_i.usedGpuMemory
        pynvml.nvmlShutdown()  # 最后关闭管理工具
        if gpu_memory_used > maxmem:
            return int(gpu_memory_used - maxmem)

    shape = c_array(ctypes.c_int64, shape)
    ndim = ctypes.c_int(len(shape))
    handle = DLArrayHandle()

    memorytoSaving = ctypes.c_int(0)
    memorytoSaving = ctypes.pointer(memorytoSaving)
    _LIB.DLArrayAlloc(shape, ndim, ctx, ctypes.byref(handle),memorytoSaving)
    memorytoSaving = _LIB.getInt(memorytoSaving)
    if memorytoSaving == 0:
        return NDArray(handle)
    else:
        return memorytoSaving



#这下面的都要用isinstance==int
def Normalex(loc=0,scale=1,size=None ,ctx=cpu(0)):
    arr=np.random.normal(loc,scale,size)
    b0= np.random.normal(loc, scale,size[0])
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if not isinstance(b0,np.ndarray):
        b0 =np.array(b0)
    w = empty(arr.shape,ctx)
    if isinstance(w,int):
        return w
    b = empty(arr.shape, ctx)
    if isinstance(b,int):
        return b
    w._sync_copyfrom(arr)
    b._sync_copyfrom(b0)
    return w,b

def Uniformex(low=0,high=1,size=None,ctx=cpu(0)):
    arr =np.random.uniform(low,high, size)
    b0 = np.random.uniform(low,high,size[0])
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if not isinstance(b0, np.ndarray):
        b0 = np.array(b0)
    w = empty(arr.shape, ctx)
    if isinstance(w, int):
        return w
    b = empty(arr.shape, ctx)
    if isinstance(b, int):
        return b
    w._sync_copyfrom(arr)
    b._sync_copyfrom(b0)
    return w, b

def xavier_Normalex(loc=0,size=None ,ctx=cpu(0)):
    scale=np.sqrt(2.0/(size[0]+size[1]))
    arr = np.random.normal(loc, scale, size)
    b0 = np.random.normal(loc, scale, size[0])
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if not isinstance(b0, np.ndarray):
        b0 = np.array(b0)
    w = empty(arr.shape, ctx)
    if isinstance(w, int):
        return w
    b = empty(arr.shape, ctx)
    if isinstance(b, int):
        return b
    w._sync_copyfrom(arr)
    b._sync_copyfrom(b0)
    return w, b

def xavier_Uniformex(size=None,ctx=cpu(0)):
    a = np.sqrt(6.0/(size[0]+size[1]))
    arr = np.random.uniform(-a, a, size)
    b0 = np.random.uniform(-a, a, size[0])
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if not isinstance(b0, np.ndarray):
        b0 = np.array(b0)
    w = empty(arr.shape, ctx)
    if isinstance(w, int):
        return w
    b = empty(arr.shape, ctx)
    if isinstance(b, int):
        return b
    w._sync_copyfrom(arr)
    b._sync_copyfrom(b0)
    return w, b

def Normal(loc=0,scale=1,size=None ,ctx=cpu(0)):
    arr=np.random.normal(loc,scale,size)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    w = empty(arr.shape, ctx)
    if isinstance(w, int):
        return w
    w._sync_copyfrom(arr)
    return w

def Uniform(low=0,high=1,size=None,ctx=cpu(0)):
    arr =np.random.uniform(low,high, size)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    w = empty(arr.shape, ctx)
    if isinstance(w, int):
        return w
    w._sync_copyfrom(arr)
    return w

def xavier_Normal(loc=0,size=None ,ctx=cpu(0)):
    scale=np.sqrt(2.0/(size[0]+size[1]))
    arr = np.random.normal(loc, scale, size)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    w = empty(arr.shape, ctx)
    if isinstance(w, int):
        return w
    w._sync_copyfrom(arr)
    return w

def xavier_Uniform(size=None,ctx=cpu(0)):
    a = np.sqrt(6.0/(size[0]+size[1]))
    arr = np.random.uniform(-a, a, size)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    w = empty(arr.shape, ctx)
    if isinstance(w, int):
        return w
    w._sync_copyfrom(arr)
    return w