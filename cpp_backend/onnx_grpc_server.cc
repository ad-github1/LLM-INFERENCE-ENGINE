#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "inference.grpc.pb.h"
#include "onnx_runtime_core.h"

namespace {

class InferenceBackendService final : public llmengine::InferenceBackend::Service {
 public:
  explicit InferenceBackendService(std::string model_path) : core_(std::move(model_path)) {}

  grpc::Status Prefill(
      grpc::ServerContext* context,
      const llmengine::PrefillRequest* request,
      llmengine::ModelOutputBatch* response) override {
    (void)context;
    std::vector<NativePrefillItem> items;
    items.reserve(request->items_size());
    for (const auto& item : request->items()) {
      NativePrefillItem native_item;
      native_item.request_id = item.request_id();
      native_item.token_start = item.token_start();
      native_item.token_ids.assign(item.token_ids().begin(), item.token_ids().end());
      native_item.is_final_chunk = item.is_final_chunk();
      items.push_back(std::move(native_item));
    }

    for (const auto& output_item : core_.Prefill(items)) {
      auto* output = response->add_items();
      output->set_request_id(output_item.request_id);
      output->set_next_token(output_item.next_token);
    }
    return grpc::Status::OK;
  }

  grpc::Status Decode(
      grpc::ServerContext* context,
      const llmengine::DecodeRequest* request,
      llmengine::ModelOutputBatch* response) override {
    (void)context;
    std::vector<NativeDecodeItem> items;
    items.reserve(request->items_size());
    for (const auto& item : request->items()) {
      items.push_back(NativeDecodeItem{item.request_id(), item.token_id(), item.position()});
    }

    for (const auto& output_item : core_.Decode(items)) {
      auto* output = response->add_items();
      output->set_request_id(output_item.request_id);
      output->set_next_token(output_item.next_token);
    }
    return grpc::Status::OK;
  }

 private:
  OnnxRuntimeCore core_;
};

void RunServer(const std::string& address, const std::string& model_path) {
  InferenceBackendService service(model_path);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Inference backend listening on " << address << std::endl;
  server->Wait();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string address = argc > 1 ? argv[1] : "0.0.0.0:50051";
  const std::string model_path = argc > 2 ? argv[2] : "";
  RunServer(address, model_path);
  return 0;
}
