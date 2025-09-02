import onnx

model = onnx.load("models/best/best.onnx")
print([input.name for input in model.graph.input])
print([output.name for output in model.graph.output])

