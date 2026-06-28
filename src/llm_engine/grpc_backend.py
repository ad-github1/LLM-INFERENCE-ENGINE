from __future__ import annotations

from llm_engine.backend import ModelBackend, ModelOutput
from llm_engine.types import DecodeWork, PrefillWork


class GrpcModelBackend(ModelBackend):
    """Optional client for a C++/ONNX Runtime backend exposed over gRPC.

    The generated protobuf modules are intentionally optional so the simulator
    remains runnable with the Python standard library only.
    """

    def __init__(self, target: str) -> None:
        try:
            import grpc  # type: ignore[import-not-found]
            import inference_pb2  # type: ignore[import-not-found]
            import inference_pb2_grpc  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "GrpcModelBackend requires grpcio plus generated modules from "
                "proto/inference.proto. Install grpcio/grpcio-tools and generate "
                "inference_pb2.py + inference_pb2_grpc.py before using it."
            ) from exc

        self._grpc = grpc
        self._pb2 = inference_pb2
        self._channel = grpc.insecure_channel(target)
        self._stub = inference_pb2_grpc.InferenceBackendStub(self._channel)

    def prefill(self, work: tuple[PrefillWork, ...]) -> tuple[ModelOutput, ...]:
        items = [
            self._pb2.PrefillItem(
                request_id=item.request_id,
                token_start=item.token_start,
                token_ids=list(item.tokens),
                is_final_chunk=item.is_final_chunk,
            )
            for item in work
        ]
        response = self._stub.Prefill(self._pb2.PrefillRequest(items=items))
        return tuple(ModelOutput(request_id=item.request_id, next_token=item.next_token) for item in response.items)

    def decode(self, work: tuple[DecodeWork, ...]) -> tuple[ModelOutput, ...]:
        items = [
            self._pb2.DecodeItem(
                request_id=item.request_id,
                token_id=item.token_id,
                position=item.position,
            )
            for item in work
        ]
        response = self._stub.Decode(self._pb2.DecodeRequest(items=items))
        return tuple(ModelOutput(request_id=item.request_id, next_token=item.next_token) for item in response.items)
