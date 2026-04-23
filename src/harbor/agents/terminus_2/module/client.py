from typing import List, Dict, Optional, Any
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


# Optional named API configs.
#
# TACO never requires this dict to be populated: pass `compress_base_url` /
# `compress_api_key` / `compress_model_name` directly from the launch script
# (see scripts/tb_eval.sh for an example) and LLMClient will be instantiated
# without looking up any named config.
#
# If you prefer to keep a couple of reusable profiles, add them here or at
# runtime via `LLMClient.add_api_config(...)`. Example:
#
# API_CONFIGS = {
#     "openai": {
#         "base_url": "https://api.openai.com/v1",
#         "api_key": os.environ.get("OPENAI_API_KEY", ""),
#         "model_name": "gpt-4o-mini",
#         "use_completions": False,
#     },
# }
API_CONFIGS: Dict[str, Dict[str, Any]] = {}

class LLMClient:
    """
    LLM client for making chat completion requests using OpenAI SDK format.
    
    Supports:
    - Custom base_url for OpenAI-compatible API endpoints
    - API key authentication
    - System and user prompt configuration
    - API configuration dictionary for easy switching between different APIs
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 0,
        api_config_key: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            base_url: Base URL for the OpenAI-compatible API endpoint (ignored if api_config_key is provided)
            api_key: API key for authentication (optional, ignored if api_config_key is provided)
            model_name: Default model name to use (optional, can be set per request)
            system_prompt: Default system prompt (optional)
            user_prompt: Default user prompt template (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries (0 means no retries)
            api_config_key: Key to use from API_CONFIGS dictionary (if provided, overrides base_url and api_key)
        """
        if api_config_key:
            config = LLMClient.get_api_config(api_config_key)
            self.base_url = config["base_url"]
            self.api_key = config["api_key"] or "sk-no-key-required"
            self.model_name = config.get("model_name") or model_name
            self.use_completions = config.get("use_completions", False)
        else:
            if base_url is None:
                raise LLMClientError("Either base_url or api_config_key must be provided")
            self.base_url = base_url.rstrip('/')
            self.api_key = api_key or "sk-no-key-required"  # default for keyless endpoints
            self.model_name = model_name
            self.use_completions = False
        
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize OpenAI client with custom base_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
    
    def set_base_url(self, base_url: str) -> None:
        """Set the base URL for API calls and recreate the OpenAI client."""
        self.base_url = base_url.rstrip('/')
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
    
    def set_api_key(self, api_key: str) -> None:
        """Set the API key for authentication and recreate the OpenAI client."""
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the default system prompt."""
        self.system_prompt = system_prompt
    
    def set_user_prompt(self, user_prompt: str) -> None:
        """Set the default user prompt template."""
        self.user_prompt = user_prompt
    
    def set_model_name(self, model_name: str) -> None:
        """Set the default model name."""
        self.model_name = model_name
    
    @classmethod
    def add_api_config(
        cls,
        key: str,
        base_url: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        """Register (or overwrite) a named API config in ``API_CONFIGS``.

        Args:
            key: config name.
            base_url: API base URL.
            api_key: API key (optional).
            model_name: default model name (optional).
        """
        API_CONFIGS[key] = {
            "base_url": base_url.rstrip('/'),
            "api_key": api_key or "sk-no-key-required",
            "model_name": model_name
        }
    
    @classmethod
    def get_api_config(cls, key: str) -> Dict[str, Any]:
        """Return a copy of the API config registered under ``key``.

        Raises:
            LLMClientError: if ``key`` is not registered in ``API_CONFIGS``.
        """
        if key not in API_CONFIGS:
            raise LLMClientError(
                f"API config '{key}' is not registered. Available keys: {list(API_CONFIGS.keys())}"
            )
        return API_CONFIGS[key].copy()
    
    @classmethod
    def list_api_configs(cls) -> List[str]:
        """Return the list of registered API config keys."""
        return list(API_CONFIGS.keys())
    
    def switch_api(self, key: str) -> None:
        """Switch this client to the API config registered under ``key``.

        Raises:
            LLMClientError: if ``key`` is not registered.
        """
        config = LLMClient.get_api_config(key)
        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        if config.get("model_name"):
            self.model_name = config["model_name"]
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Flatten an OpenAI-style chat message list into a plain-text prompt
        suitable for the ``/v1/completions`` endpoint (no chat template)."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def chat(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_content: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        use_chat_completions: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Make a chat completion request using OpenAI SDK format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                      If None, will construct from system_prompt and user_content.
            model: Model name to use (defaults to self.model_name)
            system_prompt: System prompt to use (defaults to self.system_prompt)
            user_content: User content to use (defaults to self.user_prompt)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in completion (None for model default)
            reasoning_effort: Reasoning effort (None for model default)
            use_chat_completions: If None, fall back to ``self.use_completions``.
                                 If True, use /v1/chat/completions endpoint (requires chat template).
                                 If False, use /v1/completions endpoint (no chat template needed).
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            str: The response content from the API
        
        Raises:
            LLMClientError: If the API call fails
        """
        # Determine which model to use
        model_to_use = model or self.model_name
        if not model_to_use:
            raise LLMClientError("Model name must be specified either in constructor or in chat() call")
        
        # Pick endpoint: explicit arg wins, otherwise fall back to self.use_completions.
        use_completions_endpoint = self.use_completions if use_chat_completions is None else not use_chat_completions
        
        # Construct messages if not provided
        if messages is None:
            messages = []
            
            # Add system prompt if available
            system_content = system_prompt or self.system_prompt
            if system_content:
                messages.append({"role": "system", "content": system_content})
            
            # Add user content if available
            user_content_to_use = user_content or self.user_prompt
            if user_content_to_use:
                messages.append({"role": "user", "content": user_content_to_use})
            
            if not messages:
                raise LLMClientError("Either messages or system_prompt/user_content must be provided")
        
        # Prepare request parameters
        request_params = {
            'model': model_to_use,
            'messages': messages,
            'temperature': temperature,
        }
        
        # Only add reasoning_effort if specified (some models don't support it)
        if reasoning_effort is not None:
            request_params['reasoning_effort'] = reasoning_effort
        
        if max_tokens is not None:
            request_params['max_tokens'] = max_tokens
        
        # Add any additional kwargs
        request_params.update(kwargs)
        
        try:
            if use_completions_endpoint:
                # /v1/completions endpoint (no chat template on the server side).
                prompt = self._messages_to_prompt(request_params['messages'])

                completions_params = {
                    'model': request_params['model'],
                    'prompt': prompt,
                    'temperature': request_params['temperature'],
                }

                if 'max_tokens' in request_params:
                    completions_params['max_tokens'] = request_params['max_tokens']

                # Forward remaining kwargs (except chat-only ones).
                for key, value in kwargs.items():
                    if key not in ['messages', 'reasoning_effort']:
                        completions_params[key] = value
                
                response = self.client.completions.create(**completions_params)
                
                # Extract response content
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    content = getattr(choice, 'text', '')
                    return content.strip()
                else:
                    raise LLMClientError("Invalid response format: no choices in response")
            else:
                # /v1/chat/completions endpoint (server applies chat template).
                response = self.client.chat.completions.create(**request_params)
                
                # Extract response content
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and choice.message:
                        content = getattr(choice.message, 'content', '')
                        return content
                    else:
                        raise LLMClientError("Invalid response format: message not found")
                else:
                    raise LLMClientError("Invalid response format: no choices in response")
                
        except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as e:
            error_msg = f"API error: {e}"
            raise LLMClientError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            raise LLMClientError(error_msg) from e
    
    def chat_with_usage(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_content: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> tuple[str, dict]:
        """
        Make a chat completion request and return both content and usage info.
        
        Returns:
            tuple[str, dict]: (response_content, usage_info)
            usage_info contains: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        """
        model_to_use = model or self.model_name
        if not model_to_use:
            raise LLMClientError("Model name must be specified")
        
        # Construct messages if not provided
        if messages is None:
            messages = []
            system_content = system_prompt or self.system_prompt
            if system_content:
                messages.append({"role": "system", "content": system_content})
            user_content_to_use = user_content or self.user_prompt
            if user_content_to_use:
                messages.append({"role": "user", "content": user_content_to_use})
            if not messages:
                raise LLMClientError("Either messages or system_prompt/user_content must be provided")
        
        request_params = {
            'model': model_to_use,
            'messages': messages,
            'temperature': temperature,
        }
        if max_tokens is not None:
            request_params['max_tokens'] = max_tokens
        request_params.update(kwargs)
        
        try:
            response = self.client.chat.completions.create(**request_params)
            
            # Extract content
            content = ""
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    content = getattr(choice.message, 'content', '')
            
            # Extract usage
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(response, 'usage') and response.usage:
                usage["prompt_tokens"] = getattr(response.usage, 'prompt_tokens', 0) or 0
                usage["completion_tokens"] = getattr(response.usage, 'completion_tokens', 0) or 0
                usage["total_tokens"] = getattr(response.usage, 'total_tokens', 0) or 0
            
            return content, usage
            
        except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as e:
            raise LLMClientError(f"API error: {e}") from e
        except Exception as e:
            raise LLMClientError(f"Unexpected error: {e}") from e
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"LLMClient(base_url='{self.base_url}', "
            f"model='{self.model_name}', "
            f"has_api_key={self.api_key is not None})"
        )


if __name__ == "__main__":
    # Minimal smoke test. Configure via environment variables:
    #   LLMCLIENT_TEST_BASE_URL, LLMCLIENT_TEST_API_KEY, LLMCLIENT_TEST_MODEL
    import os
    import traceback

    base_url = os.environ.get("LLMCLIENT_TEST_BASE_URL")
    api_key = os.environ.get("LLMCLIENT_TEST_API_KEY")
    model = os.environ.get("LLMCLIENT_TEST_MODEL")

    if not base_url or not model:
        print(
            "Set LLMCLIENT_TEST_BASE_URL and LLMCLIENT_TEST_MODEL to run the smoke-test."
        )
    else:
        try:
            client = LLMClient(base_url=base_url, api_key=api_key, model_name=model)
            print(f"Client ready: {client}")
            response = client.chat(
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=8,
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Smoke-test failed: {e}")
            traceback.print_exc()
