Flash-Attention-2 with GPT-OSS style Offset (Sinks)
===========================

[GPT-OSS](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf) have released and one notable feature is that each attention head now includes a learned bias term in the denominator of the softmax. This is similar to techniques like [off-by-one attention](https://www.evanmiller.org/attention-is-off-by-one.html) and [attention sinks](https://arxiv.org/abs/2309.17453).


For the ease of understanding and implementation, in this blog I will modify the [official triton FA2 tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) to add this function.
Check the explaination in Blog: [https://zirui-ray-liu.github.io/blog/2025/FA+Offset/](https://zirui-ray-liu.github.io/blog/2025/FA+Offset/)

It requires **triton 3.4.0**

## FA2 FWD Change

L227, 228 in fa.py

## FA2 BWD Change
L483-487 in fa.py

## Note

This implementation is not the most hardware-efficient one. But it is easy to understand and can be implemented with minimal line of changes.
