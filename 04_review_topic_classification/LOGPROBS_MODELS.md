# Models That Return Logprobs From the API

To get **real token-level log probabilities** (not just JSON `topic_probabilities`), the model and endpoint must support the `logprobs` parameter in the Chat Completions response.

## Bedrock and logprobs

**Yes – Bedrock can return logprobs for output tokens**, but only in one case:

- **Custom Model Import (CMI)**  
  For models **imported** into Bedrock after Nov 11, 2025, you use **InvokeModel** (not Converse) with:
  - **BedrockCompletion**: `return_logprobs: true` → logprobs for **output tokens** only.
  - **OpenAIChatCompletion**: `logprobs: true`, `top_logprobs`: N (and optionally `prompt_logprobs`) → same shape as OpenAI (per-token `token`, `logprob`, `top_logprobs`).

- **Native foundation models (Claude, Nova, Llama, etc.)**  
  The **Converse** API does **not** expose logprobs. Responses have no logprobs field for these models.

So: Bedrock does support “storing”/returning logprobs for output tokens when the model is an **imported** one and you call **InvokeModel** with the right payload. For native Claude (or other foundation models) via Converse, you do **not** get logprobs from the API.

## Where logprobs are supported

| Provider / Endpoint | Models | Logprobs in API? |
|--------------------|--------|-------------------|
| **OpenAI** (api.openai.com) | **gpt-4o**, **gpt-4o-mini**, gpt-4-turbo, gpt-4 | Yes – use `logprobs=True` in Chat Completions |
| **AWS Bedrock – Custom Model Import** | Your imported model (via InvokeModel) | Yes – BedrockCompletion or OpenAIChatCompletion with `return_logprobs` / `logprobs` |
| **AWS Bedrock Converse** (native models) | anthropic.claude-*, Amazon Nova, etc. | No – Converse does not return logprobs |
| **AWS Bedrock Mantle** (OpenAI-compatible) | openai.gpt-oss-120b, openai.gpt-oss-20b | Not documented; may return `logprobs: null` |
| **Anthropic** (direct API) | Claude | No – Messages API does not support logprobs |

## Recommendation for Stage 04 (topic classification with logprobs)

**Option A – Bedrock (same behaviour as OpenAI logprobs)**  
- Use a Bedrock model that supports logprobs in Chat Completions (e.g. **gpt-oss** on Mantle).
- Set `logprobs_model_id` (and optionally `bedrock_model_id`) to **`openai.gpt-oss-120b`** (or `openai.gpt-oss-20b`).
- Keep `OPENAI_BASE_URL` pointing at Mantle (e.g. `https://bedrock-mantle.us-west-2.api.aws/v1`).
- The script requests `logprobs=True` and `top_logprobs=3` for gpt-oss and uses the returned token logprobs when present (same as OpenAI).

**Option B – OpenAI**  
- Set `hyperparameters.model` to **`gpt-4o-mini`** or **`gpt-4o`**.
- Do **not** set `OPENAI_BASE_URL` (or set it to `https://api.openai.com/v1`).
- Use your OpenAI API key in `OPENAI_API_KEY`. The script requests logprobs and fills `theme_token_probabilities` from the API response.

## If you use Claude on Bedrock

- **Claude** (via Converse): Converse does not return API logprobs. The script uses JSON `topic_probabilities` from the model output (prompt asks for them) or fallback 0.5.

## References

- [OpenAI – Using logprobs](https://developers.openai.com/cookbook/examples/using_logprobs/)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) – `logprobs`, `top_logprobs`
- [AWS Bedrock – Advanced API features for imported models (log probabilities)](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-import-advanced-features.html) – BedrockCompletion (`return_logprobs`) and OpenAIChatCompletion (`logprobs`, `top_logprobs`, `prompt_logprobs`) for **Custom Model Import** only.
