# Structured Outlines

A minimalist, zero-dependency library for **structured LLM generation**, designed for high-security environments and restricted platforms like Google Colab.

## 🌟 Why Structured Outlines?

In many professional or academic environments, installing external dependencies is restricted due to security protocols. **Structured Outlines** was built to solve this by relying exclusively on the standard libraries already present in common ML environments (like `transformers`, `torch`, and `pydantic`) such as Google Colab.

### Key Strengths:
* **Auditable in Minutes:** The entire core logic is contained in under **1,000 lines of code**. You can review every single line of the regex engine and logits processor before deployment.
* **Security-First:** No hidden bloat or complex dependency trees. If you can't install third-party packages, you can simply copy these files into your project.
* **Built for Colab:** Optimized to run seamlessly in Google Colab using the pre-installed Hugging Face stack.
* **Full Transparency:** Designed for users who need to know exactly how their structured execution is performed, step-by-step.

## ⚙️ How it Works

The library ensures 100% valid JSON output by constraining the model at the token level:
1.  **Schema to Regex:** Converts Pydantic models into specialized regular expressions.
2.  **Regex to DFA:** Compiles the regex into a Deterministic Finite Automaton (DFA) using a custom, lightweight engine.
3.  **Token Masking:** A `LogitsProcessor` uses the DFA to mask out any tokens that would violate the schema during generation.

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from structured_outlines import generate

# 1. Define your structure
class UserProfile(BaseModel):
    name: str
    age: int
    is_student: bool

# 2. Setup your Hugging Face model
model_id = "your-choice-of-model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# 3. Generate structured JSON
prompt = "Extract info: Alex is 22 and currently enrolled in CS.\n"

json_output = generate(
    model=model,
    tokenizer=tokenizer,
    schema=UserProfile,
    prompt=prompt
)

print(json_output)
# Output: {"name": "Alex", "age": 22, "is_student": true}
```

## 📂 Project Structure
* `json_schema.py`: Pydantic to Regex conversion (~195 lines).
* `regex_parser.py`: Custom Regex to DFA engine (~485 lines).
* `guide.py`: DFA-based token transition mapping (~95 lines).
* `generate.py`: Hugging Face integration and Logits processing (~140 lines).

## ✅ Schema Features

Only **structural** constraints are enforced at the DFA level during generation. Value-level constraints are checked **after** generation by `pydantic.model_validate_json`, which raises if the output is out of bounds.

**Enforced by the DFA (token-level guarantee):**
* Types: `str`, `int`, `float`, `bool`, `None`
* `enum`, `const`, `anyOf` / `Optional[...]` / `Union[...]`
* `array` (`List[...]`), `object` (nested `BaseModel`), `$ref` to non-recursive `$defs`
* Object property order, `required` vs optional presence

**NOT enforced by the DFA** (validated post-hoc by Pydantic only):
* String: `pattern`, `min_length`, `max_length`, `format` (email, uuid, etc.)
* Numeric: `ge`, `le`, `gt`, `lt`, `multiple_of`
* Array: `min_items`, `max_items`, `unique_items`
* Number exponential notation (`1e5`) — the DFA accepts only decimal form

If your schema relies on these for correctness, expect occasional `ValidationError` at the end of generation. For a batch of N items, plan to retry or drop failures. If you need these guarantees at the token level, this library is not the right tool.

## ⚠️ Disclaimer — When NOT to Use This Library

**Structured Outlines** is intentionally small and trusting. It was designed for environments where a developer is the only user, dependencies are restricted, and auditability beats feature completeness. **Do not use it in the following scenarios:**

* **User-facing / multi-tenant systems.** The library assumes the `schema` and `prompt` come from a trusted developer. It is not hardened against adversarially crafted schemas (deeply nested `anyOf`, pathological `$ref` graphs) beyond the default DFA state cap. Production APIs exposed to end users should use a mature library (`outlines`, `guidance`, `lm-format-enforcer`, or vendor-native structured output).
* **High-throughput production inference.** `RegexGuide` precomputes transitions for every (state, token) pair. For large vocabularies (>100k) combined with complex schemas, startup cost and memory footprint are significant. There is no caching across calls.
* **Schemas requiring value-level validation at the token level.** See *Schema Features* above. If `pattern`, length bounds, or numeric ranges must be guaranteed during decoding (not after), use a library that supports them natively.
* **Beam search decoding.** Only `batch_size == 1` with greedy or sampling is supported. Beam search is rejected explicitly because only the first beam would be constrained.
* **Recursive schemas.** Regular languages cannot describe recursive grammars. Self-referencing `$ref` cycles are rejected with a clear error — restructure or bound the depth.
* **Safety-critical outputs.** Structural validity is not semantic correctness. A schema-valid JSON can still be factually wrong, toxic, or leak data. Apply the usual LLM output review on top.

If your use case is *"I need to classify 10k texts tonight on Colab without installing anything my security team hasn't already vetted"* — this library is for you. If it isn't, pick a production-grade alternative.

## 📝 License
Distributed under the **Apache License 2.0**.
