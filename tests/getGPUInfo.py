import pynvml
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(3)
# print(pynvml.nvmlDeviceGetPcieThroughput(handle, 0))
# print(pynvml.nvmlDeviceGetPcieThroughput(handle, 1))
# while True:
#     print(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
# print(pynvml.nvmlDeviceGetPerformanceState(handle))
def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(os.environ["CUDA_VISIBLE_DEVICES"])

print(os.environ["CUDA_VISIBLE_DEVICES"])
test()
print(os.environ["CUDA_VISIBLE_DEVICES"])
