# API Reference

Regent exposes a REST API on port 8400 with two interfaces: the native Regent API and an OpenAI-compatible endpoint.

---

## Starting the server

```bash
./start.sh \
    --config configs/regent_7b.yaml \
    --model checkpoints/phase1/step_50000.safetensors \
    --tokenizer data/tokenizer/regent.model
```

Or directly:

```bash
PYTHONPATH=. python3 serve/server.py \
    --config configs/regent_7b.yaml \
    --model checkpoints/phase1/step_50000.safetensors \
    --tokenizer data/tokenizer/regent.model \
    --port 8400
```

---

## Native API

### POST /generate

Generate text with verification-gated decoding, thinking, and tool calling.

**Request**

```json
{
  "messages": [
    {"role": "system", "content": "You are a legal research assistant."},
    {"role": "user", "content": "What are the elements of negligence?"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "verification": true,
  "grounding_threshold": 0.4,
  "session_id": null
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| messages | list[dict] | required | Conversation messages. Each has `role` and `content`. |
| tools | list[ToolDefinition] | null | Tool definitions the model can call. |
| epg_nodes | list[EPGNode] | null | Knowledge graph nodes to inject as context. |
| essence | EssenceState | null | Behavioral state vector. |
| max_tokens | int | 2048 | Maximum tokens to generate. |
| temperature | float | 0.7 | Sampling temperature. |
| top_p | float | 0.9 | Nucleus sampling threshold. |
| verification | bool | true | Enable per-token grounding scores. |
| grounding_threshold | float | 0.4 | Below this score, the model enters caution or halt mode. |
| session_id | string | null | Resume a previous session. Server maintains Mamba state between requests. |

**Response**

```json
{
  "text": "The four elements of negligence are duty, breach, causation, and damages.",
  "tokens": [464, 1717, 3168, ...],
  "token_texts": ["The", " four", " elements", ...],
  "grounding_scores": [0.92, 0.88, 0.91, ...],
  "halted_positions": [],
  "total_tokens": 42,
  "inference_time_ms": 312.4,
  "session_id": "a1b2c3d4-...",
  "thinking": "",
  "tool_calls": [],
  "stop_reason": "eos"
}
```

| Field | Type | Description |
|---|---|---|
| text | string | Generated text. |
| tokens | list[int] | Token IDs. |
| token_texts | list[string] | Each token decoded individually. Aligns 1:1 with grounding_scores. |
| grounding_scores | list[float] | Per-token confidence, 0 to 1. Only for visible output tokens. |
| halted_positions | list[int] | Token positions where the model halted and retrieved context. |
| total_tokens | int | Number of output tokens. |
| inference_time_ms | float | Wall-clock inference time. |
| session_id | string | Use this to continue the conversation. |
| thinking | string | Internal reasoning the model produced before answering. Empty if no thinking occurred. |
| tool_calls | list[ToolCallResponse] | Tool invocations. Non-empty only when stop_reason is "tool_call". |
| stop_reason | string | One of: "eos", "tool_call", "max_tokens". |

---

### Thinking

When the model encounters a question that requires reasoning, it thinks internally before responding. The thinking content is returned in the `thinking` field and is not included in `text`, `tokens`, or `grounding_scores`.

Thinking is triggered by the model itself during generation. There is no request parameter to force it. The model will only produce thinking if it was trained on examples containing `[THINK]...[/THINK]` blocks in Phase 2 fine-tuning data.

```json
{
  "text": "The statute of limitations for medical malpractice in California is 3 years.",
  "thinking": "The user is asking about California medical malpractice. CCP 340.5 sets the limit at 3 years from injury or 1 year from discovery, whichever comes first. The general answer is 3 years.",
  "grounding_scores": [0.95, 0.91, 0.88, ...],
  "stop_reason": "eos"
}
```

---

### Tool calling

Define tools in the request. When the model decides it needs external information, it stops generation and returns a tool call.

**Step 1: Send the request with tools**

```json
POST /generate
{
  "messages": [
    {"role": "user", "content": "What is the weather in Lagos today?"}
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Returns current weather for a given city.",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"}
        },
        "required": ["city"]
      }
    }
  ],
  "session_id": "sess-001"
}
```

**Step 2: Model returns a tool call**

```json
{
  "text": "",
  "stop_reason": "tool_call",
  "tool_calls": [
    {"name": "get_weather", "arguments": {"city": "Lagos"}}
  ],
  "session_id": "sess-001"
}
```

**Step 3: Execute the tool and send the result back**

Post to the same session. The server carries the full model state forward, so the model continues where it left off without re-encoding the conversation.

```json
POST /generate
{
  "messages": [
    {"role": "tool_result", "content": "{\"temperature\": 31, \"condition\": \"sunny\"}"}
  ],
  "session_id": "sess-001"
}
```

**Step 4: Model responds with the final answer**

```json
{
  "text": "The weather in Lagos today is 31 degrees and sunny.",
  "stop_reason": "eos",
  "tool_calls": [],
  "session_id": "sess-001"
}
```

The model will only emit tool calls if it was trained on examples containing `[TOOL_CALL]{"name": "...", "arguments": {...}}[TOOL_END]` blocks in Phase 2 fine-tuning data.

---

### Multi-turn conversations

Use `session_id` to maintain state across requests. The server persists the Mamba layer cache between requests, so the model carries forward its full internal state without re-processing the conversation history.

```json
POST /generate
{"messages": [{"role": "user", "content": "What is tort law?"}]}

// Response includes session_id: "abc-123"

POST /generate
{
  "messages": [{"role": "user", "content": "Give me an example."}],
  "session_id": "abc-123"
}
```

Sessions expire after 1 hour of inactivity. Delete a session explicitly:

```
DELETE /session/{session_id}
```

List active sessions:

```
GET /sessions
```

---

### Knowledge graph input (EPG nodes)

Inject structured knowledge as first-class model input rather than pasting text into the prompt.

```json
POST /generate
{
  "messages": [{"role": "user", "content": "What should I prescribe?"}],
  "epg_nodes": [
    {
      "key": "patient_allergy",
      "value": "Penicillin allergy confirmed 2024-01-15",
      "confidence": 0.95,
      "activation": 0.8,
      "category": "domain"
    },
    {
      "key": "drug_interaction",
      "value": "Amoxicillin contraindicated with penicillin allergy",
      "confidence": 0.99,
      "activation": 0.9,
      "category": "constraint"
    }
  ]
}
```

**EPGNode fields**

| Field | Type | Default | Description |
|---|---|---|---|
| key | string | required | Node identifier. |
| value | string | required | Node content. |
| confidence | float | 0.5 | How reliable this information is (0 to 1). |
| activation | float | 0.5 | How recently relevant this information is (0 to 1). |
| valence | float | 0.0 | Emotional valence (-1 to 1). |
| emotional_weight | float | 0.5 | How emotionally significant (0 to 1). |
| category | string | "domain" | One of: identity, belief, capability, experience, goal, domain, relationship, emotional, procedural, episodic, semantic, preference, constraint, meta, other. |

When grounding drops below the threshold and the model halts, it re-ranks available EPG nodes by confidence times activation and retrieves the most relevant ones before continuing.

---

### Verify existing text

Score existing text for grounding without generating anything new.

```json
POST /verify
{
  "text": "Aspirin reduces fever and is commonly used as a blood thinner.",
  "epg_nodes": [...]
}
```

**Response**

```json
{
  "grounding_scores": [0.95, 0.92, 0.88, 0.91, 0.45, 0.38, 0.41, ...],
  "mean_grounding": 0.72,
  "flagged_spans": [
    {
      "start": 4,
      "end": 7,
      "text": "commonly used as a blood thinner",
      "min_score": 0.38
    }
  ]
}
```

---

### Essence state

Control the model's behavioral parameters. All fields are optional and have sensible defaults.

```json
POST /generate
{
  "messages": [{"role": "user", "content": "Should I invest in this?"}],
  "essence": {
    "essence_index": 5.0,
    "truth_vs_lie": 0.0,
    "curiosity": 0.5,
    "self_preservation": 0.3
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| essence_index | float | 5.0 | Overall essence level. |
| essence_influence | float | 0.0 | How much essence affects output. |
| truth_vs_lie | float | 0.0 | Bias toward truthfulness (-1 to 1). |
| civility_vs_unruliness | float | 0.0 | Communication style (-1 to 1). |
| good_vs_evil | float | 0.0 | Moral alignment (-1 to 1). |
| curiosity | float | 0.5 | Tendency to explore vs. stay focused. |
| self_preservation | float | 0.3 | Tendency to avoid risky outputs. |

---

### Utility endpoints

```
GET /health              — {"status": "ok", "model_loaded": true}
GET /info                — Model config, parameter counts, architecture details
GET /sessions            — List active sessions with token counts and idle time
DELETE /session/{id}     — Release a session and free its memory
GET /checkpoints         — List available checkpoint files across all phases
```

---

## OpenAI-compatible API

### POST /v1/chat/completions

Drop-in replacement for OpenAI's chat completions endpoint. Any SDK, framework, or application built for OpenAI works by changing the base URL.

**Python (OpenAI SDK)**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8400/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="regent",
    messages=[
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": "What is negligence?"}
    ],
    max_tokens=256,
    temperature=0.7
)

print(response.choices[0].message.content)
```

**curl**

```bash
curl -X POST http://localhost:8400/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "regent",
        "messages": [
            {"role": "user", "content": "What is negligence?"}
        ],
        "max_tokens": 256
    }'
```

**Response format**

```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1713600000,
  "model": "regent",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Negligence is a failure to exercise reasonable care...",
        "reasoning_content": "The user is asking about tort law basics..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 42,
    "total_tokens": 42
  }
}
```

The `reasoning_content` field contains the model's internal thinking, if any. It is only present when the model produced a thinking block.

**Tool calling (OpenAI format)**

```python
response = client.chat.completions.create(
    model="regent",
    messages=[
        {"role": "user", "content": "What is the weather in Lagos?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }
    ]
)

# response.choices[0].message.tool_calls is populated
# response.choices[0].finish_reason == "tool_calls"
```

Resume with the tool result:

```python
response = client.chat.completions.create(
    model="regent",
    messages=[
        {"role": "user", "content": "What is the weather in Lagos?"},
        {"role": "assistant", "tool_calls": [...]},
        {"role": "tool", "content": '{"temperature": 31}', "tool_call_id": "call_abc123"}
    ]
)
```

**Supported parameters**

| Parameter | Supported | Notes |
|---|---|---|
| model | yes | Any string, defaults to "regent". |
| messages | yes | system, user, assistant, tool roles. |
| tools | yes | Function-type tools only. |
| max_tokens | yes | |
| temperature | yes | |
| top_p | yes | |
| n | no | Must be 1. Returns 400 if > 1. |
| stream | no | Returns 400 if true. |
| stop | accepted | Silently ignored. |

**Differences from OpenAI**

- No streaming support.
- `n > 1` is rejected.
- `prompt_tokens` in usage is always 0 (not tracked through this endpoint).
- Sessions are not maintained across requests through this endpoint. Each request is independent. Use the native `/generate` endpoint with `session_id` for stateful multi-turn conversations.
- `reasoning_content` on the message contains Regent's thinking output. This field is absent when no thinking occurred.

---

## Message roles

| Role | Direction | Used in | Description |
|---|---|---|---|
| system | Caller to model | Both APIs | System prompt. |
| user | Caller to model | Both APIs | User message. |
| assistant | Model to caller | Both APIs | Model response. |
| tool_call | Model to caller | Native API | Model-emitted tool invocation. |
| tool_result | Caller to model | Native API | Result of tool execution. |
| tool | Caller to model | OpenAI API | OpenAI-format tool result. Mapped to tool_result internally. |

---

## Special tokens

These are reserved in the tokenizer and used by the generation engine. You do not need to use them directly unless you are preparing Phase 2 training data.

| Token | ID | Purpose |
|---|---|---|
| [PAD] | 0 | Padding. |
| [BOS] | 1 | Beginning of sequence. |
| [EOS] | 2 | End of sequence. |
| [GROUND] | 3 | Grounding trigger (Ver Head). |
| [EPG] | 4 | EPG prefix boundary. |
| [META] | 5 | Metadata marker. |
| [TOOL_CALL] | 6 | Start of tool call block. |
| [TOOL_RESULT] | 7 | Start of tool result block. |
| [TOOL_END] | 8 | End of tool call or result block. |
| [UNK] | 9 | Unknown token. |
| [THINK] | 10 | Start of thinking block. |
| [/THINK] | 11 | End of thinking block. |

**Phase 2 training data format for tool calling:**

```
<user>What is the weather in Lagos?
<assistant>[TOOL_CALL]{"name": "get_weather", "arguments": {"city": "Lagos"}}[TOOL_END]
[TOOL_RESULT]{"temperature": 31, "condition": "sunny"}[TOOL_END]
The weather in Lagos is 31 degrees and sunny.
```

**Phase 2 training data format for thinking:**

```
<user>What are the implications of this ruling?
<assistant>[THINK]The ruling in Smith v. Jones established that duty of care extends to... This means the user's situation would fall under...[/THINK]
Based on the ruling, three implications apply to your case...
```

---

## Verification-gated decoding

The generation engine operates in three zones based on the per-token grounding score:

| Zone | Grounding score | Behavior |
|---|---|---|
| FLOW | > 0.6 | Normal sampling at the configured temperature. |
| CAUTION | 0.3 to 0.6 | Lower temperature, conservative token selection. |
| HALT | < 0.3 | Stop. Retrieve context from EPG nodes. Restart from the uncertain point. |

The `grounding_threshold` parameter in the request controls where CAUTION begins. The HALT threshold is fixed at 0.3. HALT retrieval is limited to one retry per generation to prevent loops.

When HALT fires and no EPG nodes are available, the model falls back to CAUTION sampling rather than stopping entirely.

---

## Error handling

All endpoints return standard HTTP status codes:

| Code | Meaning |
|---|---|
| 200 | Success. |
| 400 | Bad request. Streaming requested, n > 1, or malformed input. |
| 503 | Model not loaded. Start the server with --model and --tokenizer. |
