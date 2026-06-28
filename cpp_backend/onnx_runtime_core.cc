#include "onnx_runtime_core.h"

#include <utility>

#ifdef LLM_ENGINE_ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

OnnxRuntimeCore::OnnxRuntimeCore(std::string model_path) : model_path_(std::move(model_path)) {
#ifdef LLM_ENGINE_ENABLE_ONNXRUNTIME
  // A production implementation would initialize Ort::Env, Ort::SessionOptions,
  // and Ort::Session here, then bind request block tables to model inputs.
#endif
}

std::vector<NativeModelOutput> OnnxRuntimeCore::Prefill(const std::vector<NativePrefillItem>& items) {
  std::vector<NativeModelOutput> outputs;
  outputs.reserve(items.size());
  for (const auto& item : items) {
    if (!item.is_final_chunk) {
      continue;
    }
    const std::int32_t seed = item.token_ids.empty() ? 0 : item.token_ids.back();
    outputs.push_back(
        NativeModelOutput{item.request_id, NextToken(seed, item.token_start + static_cast<std::int32_t>(item.token_ids.size()))});
  }
  return outputs;
}

std::vector<NativeModelOutput> OnnxRuntimeCore::Decode(const std::vector<NativeDecodeItem>& items) {
  std::vector<NativeModelOutput> outputs;
  outputs.reserve(items.size());
  for (const auto& item : items) {
    outputs.push_back(NativeModelOutput{item.request_id, NextToken(item.token_id, item.position)});
  }
  return outputs;
}

std::int32_t OnnxRuntimeCore::NextToken(std::int32_t seed, std::int32_t position) const {
  constexpr std::int32_t kVocabSize = 32000;
  const std::int64_t token =
      (static_cast<std::int64_t>(seed) * 1103515245 + static_cast<std::int64_t>(position) * 12345 + 17) %
      kVocabSize;
  return static_cast<std::int32_t>(token);
}
