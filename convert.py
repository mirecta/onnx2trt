#!/usr/bin/python3
# Build TensorRT engine from ONNX saved model and serialize engine to file 

import tensorrt as trt
import sys
import os




TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def printBindings(engine):
    for i in range(0, engine.num_bindings):
        if engine.binding_is_input(i):
            print("Input {} shape is {} ".format(i,engine.get_binding_shape(i)))
        else:
            print("Output {} shape is {} ".format(i,engine.get_binding_shape(i) ))


def buildEngine( onnx_file_path,engine_file_path ):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please first generate it.'.format(onnx_file_path))
                sys.exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            print('Network has {} layers'.format(network.num_layers))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            printBindings(engine)

            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = input_file.replace('.onnx','.engine')
    engine = buildEngine(input_file, output_file)
    



