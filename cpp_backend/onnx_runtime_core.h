#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct NativeModelOutput {
  std::string request_id;
  std::int32_t next_token;
};

struct NativePrefillItem {
  std::string request_id;
  std::int32_t token_start;
  std::vector<std::int32_t> token_ids;
  bool is_final_chunk;
};

struct NativeDecodeItem {
  std::string request_id;
  std::int32_t token_id;
  std::int32_t position;
};

class OnnxRuntimeCore {
 public:
  explicit OnnxRuntimeCore(std::string model_path = "");

  std::vector<NativeModelOutput> Prefill(const std::vector<NativePrefillItem>& items);
  std::vector<NativeModelOutput> Decode(const std::vector<NativeDecodeItem>& items);

 private:
  std::string model_path_;

  std::int32_t NextToken(std::int32_t seed, std::int32_t position) const;
};
