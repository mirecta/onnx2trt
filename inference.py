#!/usr/bin/python3
#Read engine from file and load it into cuda device and run  

import tensorrt as trt
import sys
import os
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np 
import pickle

"""
partialy inspired by nvidia man pages and 
https://github.com/jkjung-avt/tensorrt_demos/blob/master/utils/ssd.py
#sexy piece of code :)
import numpy as np
import pickle
import cv2

img = cv2.imread('picture.jpg',0)
pimg = preprocesImg(img)
arr = pimg.ravel()
pickle.dump(arr, open("picture.plk", "wb"))

print pickle.load(open("picture.pkl"))
"""
"""
def preprocesImg(img, shape=(300, 300)):
    
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img *= (2.0/255.0)
    img -= 1.0
    #return img.ravel()
    return img
"""

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class Engine:
    def __init__(self,filename):
        """
        load engine from file
        """
        self.engine = self._loadEngine(filename)
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._createContext()
        
    
    def _createContext(self):
        """
        allocation memory and prepare it
        """
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs

    def _loadEngine(self, filename):
        with open(filename, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


    def printBindings(self):
        """
        print inputs and outputs shape
        """
        for i in range(0, self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                print("Input {} shape is {} ".format(i,self.engine.get_binding_shape(i)))
            else:
                print("Output {} shape is {} ".format(i,self.engine.get_binding_shape(i)))

    def execute(self,input):
        """
        execute engine
        """
        np.copyto(self.host_inputs[0], input)
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        
        for i in range(0,len(self.cuda_outputs)):
            cuda.memcpy_dtoh_async(
                self.host_outputs[i], self.cuda_outputs[i], self.stream)
        self.stream.synchronize()

        return self.host_outputs



if __name__ == "__main__":
    input_file = sys.argv[1]
    engine = Engine(input_file)
    engine.printBindings()

    arr = pickle.load(open("picture.plk",'rb'))
    clases, boxes = engine.execute(arr)
    print(clases.shape)
    clases = clases.reshape((1,3000,21))
    boxes = boxes.reshape((1,3000,4))
    #label 15 is person
    for i  in range(0,len(clases[0])):
        if clases[0][i][15] >= 0.85:
            print("person detected {}".format(clases[0][i][15]))
            print ("bbox {}".format(boxes[0][i]))    
    