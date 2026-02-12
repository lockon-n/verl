import openai
import base64

# base_url="https://api.moonshot.cn/v1"
base_url="http://localhost:21100/v1"
# base_url="http://localhost:8100/v1"
# api_key="sk-HUMECC0cIzeKlbC69RU504WXJO1AZgcagBYFsGLYXWPJ6W1v"
api_key="EMPTY"

model_name="Qwen3-VL-8B-Instruct"
# model_name="kimi-k2.5"

image_path="jl_patch/ange.jpeg"
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Please describe the image in detail."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
    ],
}

client = openai.OpenAI(base_url=base_url, api_key=api_key)
response = client.chat.completions.create(
    model=model_name,
    messages=[message],
    # stream=True,  # 开启流式输出
)

# 分别打印content和reasoning_content
if hasattr(response.choices[0].message, "reasoning_content"):
    print(response.choices[0].message.reasoning_content)
    print("--------------------------------")
print(response.choices[0].message.content)