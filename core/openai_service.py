import json
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class Message:
    content: list
    stop_reason: str  # "tool_use", "end_turn" or "max_tokens"
    role: str = "assistant"


def _get(obj, key, default=None):
    """Read a field from either a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OpenAIService:
    def __init__(self, model: str):
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model

    # ---- same helper methods as the original Claude class -----------------

    def add_user_message(self, messages: list, message):
        messages.append({
            "role": "user",
            "content": message.content if isinstance(message, Message) else message,
        })

    def add_assistant_message(self, messages: list, message):
        messages.append({
            "role": "assistant",
            "content": message.content if isinstance(message, Message) else message,
        })

    def text_from_message(self, message: Message):
        return "\n".join(
            block.text for block in message.content if block.type == "text"
        )

    # ---- format conversion: Anthropic style -> OpenAI style ----------------

    def _convert_tools(self, tools):
        return [
            {
                "type": "function",
                "function": {
                    "name": _get(tool, "name"),
                    "description": _get(tool, "description") or "",
                    "parameters": _get(tool, "input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]

    def _convert_messages(self, messages, system):
        converted = []
        if system:
            converted.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Plain text message
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            if role == "assistant":
                texts, tool_calls = [], []
                for block in content:
                    block_type = _get(block, "type")
                    if block_type == "text":
                        texts.append(_get(block, "text"))
                    elif block_type == "tool_use":
                        tool_calls.append({
                            "id": _get(block, "id"),
                            "type": "function",
                            "function": {
                                "name": _get(block, "name"),
                                "arguments": json.dumps(_get(block, "input") or {}),
                            },
                        })
                assistant_msg = {"role": "assistant", "content": "\n".join(texts) or None}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                converted.append(assistant_msg)

            else:  # user message: may contain text and/or tool results
                texts = []
                for block in content:
                    block_type = _get(block, "type")
                    if block_type == "tool_result":
                        result = _get(block, "content", "")
                        if isinstance(result, list):
                            result = "\n".join(str(_get(item, "text", item)) for item in result)
                        elif not isinstance(result, str):
                            result = json.dumps(result, default=str)
                        converted.append({
                            "role": "tool",
                            "tool_call_id": _get(block, "tool_use_id"),
                            "content": result,
                        })
                    elif block_type == "text":
                        texts.append(_get(block, "text"))
                if texts:
                    converted.append({"role": "user", "content": "\n".join(texts)})

        return converted

    # ---- main call ---------------------------------------------------------

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,        # accepted for compatibility, not used by OpenAI
        thinking_budget=1024,  # accepted for compatibility, not used by OpenAI
    ) -> Message:
        params = {
            "model": self.model,
            "max_completion_tokens": 8000,
            "messages": self._convert_messages(messages, system),
        }

        # Some OpenAI models reject non-default temperature or stop sequences,
        # so only send them when they were actually changed.
        if temperature != 1.0:
            params["temperature"] = temperature
        if stop_sequences:
            params["stop"] = stop_sequences
        if tools:
            params["tools"] = self._convert_tools(tools)

        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]
        reply = choice.message

        # Convert the OpenAI reply back into Anthropic-style blocks
        blocks = []
        if reply.content:
            blocks.append(TextBlock(text=reply.content))
        for call in reply.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            blocks.append(ToolUseBlock(id=call.id, name=call.function.name, input=args))

        if reply.tool_calls:
            stop_reason = "tool_use"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        return Message(content=blocks, stop_reason=stop_reason)