# Optional C++ ONNX Runtime gRPC Backend

The Python engine is dependency-light and runs by default with `FakeModelBackend`.
This folder contains the native backend boundary for the stronger serving setup:

- `proto/inference.proto` defines the gRPC contract.
- `onnx_runtime_core.*` isolates the native model execution boundary.
- `onnx_grpc_server.cc` exposes the core over gRPC.
- `CMakeLists.txt` shows the expected C++ dependencies.

To turn this into a real transformer backend, generate C++ protobuf bindings,
link `grpc++`, link ONNX Runtime, and replace `NextToken` with model execution
that consumes request block tables from the Python scheduler.

This native backend is intentionally optional because gRPC and ONNX Runtime are
large external dependencies and are not required for the simulator tests.
