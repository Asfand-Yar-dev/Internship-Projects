from pipecat.services.openai.base_llm import BaseOpenAILLMService

class MyOpenAILLMService(BaseOpenAILLMService):
    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        params["stream"] = False  # Ensure streaming is disabled
        return params

    async def get_chat_completions(self, params_from_context):
        params = self.build_chat_completion_params(params_from_context)
        params["stream"] = False  # Double-check here as well
        # Optionally, add a debug print or logger here
        # print("DEBUG: stream param is", params["stream"])
        return await self._client.chat.completions.create(**params)