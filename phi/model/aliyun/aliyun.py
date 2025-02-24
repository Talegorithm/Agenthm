from typing import Optional, Iterator, List
from os import getenv
from dataclasses import dataclass

from phi.model.openai.like import OpenAILike
from phi.model.response import ModelResponse
from phi.model.base import Message
from openai.types.chat import ChatCompletionChunk

@dataclass
class StreamData:
    """
    用于存储流式响应数据的数据类。

    Attributes:
        response_content (str): 响应的文本内容
        response_reasoning_content (Optional[str]): 思考过程的内容
    """
    response_content: str = ""
    response_reasoning_content: Optional[str] = None

class ALIChat(OpenAILike):
    """
    A model class for aliyun Chat API.

    Attributes:
    - id: str: The unique identifier of the model. Default: "aliyun-chat".
    - name: str: The name of the model. Default: "aliyunChat".
    - provider: str: The provider of the model. Default: "aliyun".
    - api_key: Optional[str]: The API key for the model.
    - base_url: str: The base URL for the model. Default: "https://api.aliyun.com".
    """

    id: str = "deepseek-r1"
    name: str = "deepseek-r1"
    provider: str = "ALI"

    api_key: Optional[str] = getenv("ALI_API_KEY", None)
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        """
        Generate a streaming response from ALI API.
        """
        stream_data: StreamData = StreamData()
        status = ""
        # 生成响应
        for response in self.invoke_stream(messages=messages):
            if not response.choices:
                print("\nUsage:")
                print(response.usage)
                continue

            response_delta = response.choices[0].delta
            # print(response_delta)
            
            # 处理思考过程内容
            if hasattr(response_delta, 'reasoning_content') and response_delta.reasoning_content:
                if status == "":
                    status = "think"
                    response_delta.reasoning_content = "<think>" + response_delta.reasoning_content
                stream_data.response_reasoning_content = response_delta.reasoning_content
                yield ModelResponse(
                    content=response_delta.reasoning_content
                )
                
            # 处理普通内容
            if hasattr(response_delta, 'content') and response_delta.content:
                if status == "think":
                    status = "answer"
                    response_delta.content = "</think>" + response_delta.content
                stream_data.response_content += response_delta.content
                yield ModelResponse(content=response_delta.content)


    def invoke_stream(self, messages: List[Message]) -> Iterator[ChatCompletionChunk]:
        """
        Send a streaming chat completion request to the ALI API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Iterator[ChatCompletionChunk]: An iterator of chat completion chunks.
        """
        yield from self.get_client().chat.completions.create(
            model=self.id,
            messages=[m.to_dict() for m in messages],  # type: ignore
            stream=True,
            stream_options={"include_usage": True},  # 添加 stream_options 配置
            **self.request_kwargs,
        )  # type: ignore
