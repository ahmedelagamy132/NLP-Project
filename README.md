# NLP-Project
![image](https://github.com/user-attachments/assets/cd82c724-588c-46ac-9dd9-4c7950d25833)
# 🐍 LLaMA 3.2 Python Code Generator API

This project provides a RESTful API for generating **Python code** using a **fine-tuned LLaMA 3.2 1B Instruct model**, enhanced with **LoRA (Low-Rank Adaptation)**. It's ideal for applications like intelligent code assistants, developer tools, or educational platforms.

---

## 🚀 Features

- 🎯 Fine-tuned specifically for **Python code generation**
- ⚡ Powered by [Unsloth's LLaMA 3.2](https://huggingface.co/unsloth)
- 🧠 Uses **LoRA** for efficient fine-tuning and inference
- 🛠️ Simple and fast Flask API
- 🎛️ Supports decoding options: `temperature`, `top_k`, and `max_new_tokens`

---

## 📡 API Usage

### `POST /generate`

**Content-Type:** `application/json`

#### Request Body

```json
{
  "prompt": "Write a Python function to calculate factorial.",
  "max_new_tokens": 100,
  "temperature": 0.5,
  "top_k": 5
}
