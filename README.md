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

## 📝 License
Distributed under the **Apache License 2.0**.
