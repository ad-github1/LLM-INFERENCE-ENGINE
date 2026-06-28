from llm_engine import EngineConfig, InferenceEngine


def main() -> None:
    engine = InferenceEngine(
        EngineConfig(
            max_batch_size=2,
            max_prefill_tokens_per_step=4,
            cache_block_size=4,
            cache_num_blocks=8,
        )
    )
    engine.submit([101, 102, 103], max_new_tokens=4, request_id="chat-1")
    engine.submit([201, 202], max_new_tokens=3, request_id="chat-2")

    while engine.has_pending_work:
        step = engine.step()
        for event in step.token_events:
            print(f"step={step.step} {event.request_id} token={event.token_id} source={event.source}")


if __name__ == "__main__":
    main()
