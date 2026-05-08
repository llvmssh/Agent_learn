import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
]

def stream_chat(messages):
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        stream=True,
        reasoning_effort="high",
        extra_body={
            "thinking":{"type":"enabled"}
        },
    )

    reasoning_content = ""
    content = ""
    print("assistant: ", end="", flush=True)

    for chunk in response:
        if not chunk.choices:
            continue
        
        delta = chunk.choices[0].delta
        reasoning_delta = getattr(delta, "reasoning_content", None)
        content_delta = getattr(delta, "content", None)

        if reasoning_delta:
            reasoning_content += reasoning_delta

        if content_delta:
            content += content_delta
            print(content_delta, end="", flush=True)
        
    print("\n")
    return content, reasoning_content

def main():
    print("DeepSeek streaming chat started")
    print("input exit/quit out, input clear to clean context.")

    while True:
        user_input = input("user:").strip()
        if not user_input:
            continue
        
        if user_input.lower() in ("exit", "quit"):
            print("bye.")
            break
        
        if user_input.lower() == "clear":
            messages.clear()
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant.",
            })
            print("上下文已清空。\n")
            continue
        
        messages.append({"role":"user", "content":user_input})
        try:
            content, reasoning_content = stream_chat(messages)
        except Exception as e:
            print(f"\n调用失败: {e}\n")
            continue
        
        messages.append({"role": "assistant", "reasoning_content": reasoning_content, "content": content})
        
if __name__ == "__main__":
    main()