import os
import json
from openai import OpenAI
from datetime import datetime

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

SYSTEM_PROMPT = """
    You are a task-executing Agent.

    Rules:
    1. Remember previous conversation from the messages history.
    2. If the user task is simple, answer directly.
    3. If the user task is complex, first list a short plan, then execute step by step.
    4. If a tool is needed, call the appropriate tool instead of guessing.
    5. After receiving tool results, continue until you produce a final answer.
    6. Do not expose hidden reasoning. Only show concise plans and useful final answers.
"""

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date",
            "parameters": {
                "type": "object",
                "properties": {}
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply the location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name",
                    },
                    "date": {
                        "type": "string",
                        "description": "The date in format YYYY-MM-DD",
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]


def get_date_mock():
    return datetime.now().strftime("%Y-%m-%d")


def get_weather_mock(location, date):
    return f"{date} {location}: Cloudy 7~13°C"


TOOL_CALL_MAP = {
    "get_date": get_date_mock,
    "get_weather": get_weather_mock,
}


def stream_chat(messages):
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=tools,
        stream=True,
        reasoning_effort="high",
        extra_body={
            "thinking": {
                "type": "enabled"
            }
        },
    )

    reasoning_content = ""
    content = ""
    tool_call_chunks = {}

    print("assistant: ", end="", flush=True)

    tool_call_printed = False

    for chunk in response:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        reasoning_delta = getattr(delta, "reasoning_content", None)
        content_delta = getattr(delta, "content", None)
        tool_calls_delta = getattr(delta, "tool_calls", None)

        if reasoning_delta:
            reasoning_content += reasoning_delta
            # 不建议直接打印完整 reasoning_content
            # 这里打印一个点，表示模型还在生成
            print(".", end="", flush=True)

        if content_delta:
            content += content_delta
            print(content_delta, end="", flush=True)

        if tool_calls_delta:
            if not tool_call_printed:
                print("\n[tool call detected]", flush=True)
                tool_call_printed = True

            for tool_call_delta in tool_calls_delta:
                index = tool_call_delta.index

                if index not in tool_call_chunks:
                    tool_call_chunks[index] = {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    }

                current = tool_call_chunks[index]

                tool_id = getattr(tool_call_delta, "id", None)
                if tool_id:
                    # id 通常只出现一次，避免重复拼接
                    if not current["id"]:
                        current["id"] = tool_id

                tool_type = getattr(tool_call_delta, "type", None)
                if tool_type:
                    current["type"] = tool_type

                function_delta = getattr(tool_call_delta, "function", None)
                if function_delta:
                    function_name = getattr(function_delta, "name", None)
                    function_arguments = getattr(function_delta, "arguments", None)

                    if function_name:
                        current["function"]["name"] += function_name

                    if function_arguments:
                        current["function"]["arguments"] += function_arguments

    print("\n")

    tool_calls = []
    for index in sorted(tool_call_chunks.keys()):
        tool_calls.append(tool_call_chunks[index])

    if not reasoning_content and not content and not tool_calls:
        raise RuntimeError(
            "stream 结束了，但没有收到 content / reasoning_content / tool_calls。"
            "建议临时打印 raw chunk 检查返回结构。"
        )

    assistant_message = {
        "role": "assistant",
    }

    if reasoning_content:
        assistant_message["reasoning_content"] = reasoning_content

    # 有 tool_calls 时，content 可能是 None，这是正常的
    assistant_message["content"] = content if content else None

    if tool_calls:
        assistant_message["tool_calls"] = tool_calls

    return assistant_message


def run_tool_call(tool):
    function_name = tool["function"]["name"]
    raw_arguments = tool["function"]["arguments"] or "{}"

    print(f"tool name: {function_name}")
    print(f"tool arguments raw: {raw_arguments}")

    if function_name not in TOOL_CALL_MAP:
        return f"Error: unknown tool {function_name}"

    try:
        function_args = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON arguments: {e}. raw={raw_arguments}"

    tool_function = TOOL_CALL_MAP[function_name]

    try:
        return tool_function(**function_args)
    except Exception as e:
        return f"Error while running tool {function_name}: {e}"


def main():
    print("DeepSeek streaming chat started")
    print("input exit/quit out, input clear to clean context.")

    turn = 1

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
            turn = 1
            continue

        # 外层：接收用户输入
        messages.append({
            "role": "user",
            "content": user_input,
        })

        sub_turn = 1

        # 内层：处理 assistant -> tool -> assistant -> tool -> assistant
        while True:
            try:
                assistant_message = stream_chat(messages)
            except Exception as e:
                print(f"\n调用失败: {e}\n")
                break

            messages.append(assistant_message)

            reasoning_content = assistant_message.get("reasoning_content")
            content = assistant_message.get("content")
            tool_calls = assistant_message.get("tool_calls")

            print(f"Turn {turn}.{sub_turn}")
            print(f"reasoning_content exists: {bool(reasoning_content)}")
            print(f"content: {content}")
            print(f"tool_calls: {tool_calls}")

            # 没有 tool_calls，说明已经得到最终回答
            # break 只退出内层循环，然后回到 user: 等下一次输入
            if not tool_calls:
                break

            # 有 tool_calls：执行工具，并把结果写回 messages
            for tool in tool_calls:
                tool_result = run_tool_call(tool)

                print(f"tool result for {tool['function']['name']}: {tool_result}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "content": tool_result,
                })

            # 关键：这里不回到 user:
            # 而是继续内层 while，再次调用模型，让模型读取 tool result
            sub_turn += 1

        turn += 1


if __name__ == "__main__":
    main()