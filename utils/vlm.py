from io import BytesIO
from PIL import Image
import os
import base64
import random
import time
import logging
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
import numpy as np

# Set up module logging
logger = logging.getLogger(__name__)

# Import LLM logger
from utils.llm_logger import log_llm_interaction, log_llm_error

# Define the retry decorator with exponential backoff
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (Exception,),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                # Increase the delay with exponential factor and random jitter
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e
    return wrapper

class VLMBackend(ABC):
    """Abstract base class for VLM backends"""
    
    @abstractmethod
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown", system_prompt: Optional[str] = None) -> str:
        """Process an image and text prompt"""
        pass
    
    @abstractmethod
    def get_text_query(self, text: str, module_name: str = "Unknown", system_prompt: Optional[str] = None) -> str:
        """Process a text-only prompt"""
        pass

class VertexBackend(VLMBackend):
    """Google Gemini API with Vertex backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Google Generative AI package not found.")
        
        self.model_name = model_name
        
        # Initialize the model
        self.client = genai.Client(
            vertexai=True,
            project='pokeagent-012',
            location='us-central1',
        )
        self.genai = genai
        
        logger.info(f"Gemini backend initialized with model: {model_name}")
    
    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff."""
        # Vertex client in genai library requires content_parts to be the only positional argument
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=content_parts
        )
        return response
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown", system_prompt: Optional[str] = None) -> str:
        """Process an image and text prompt using Gemini API"""
        try:
            start_time = time.time()
            image = self._prepare_image(img)
            
            # Prepare content for Gemini
            content_parts = [text, image]
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                # Assuming finish_reason 12 is content policy
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name, system_prompt)
            
            result = response.text
            # Log the interaction
            duration = time.time() - start_time
            log_llm_interaction(
                interaction_type=f"vertex_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "vertex", "has_image": True, "system_prompt": bool(system_prompt)},
                model_info={"model": self.model_name, "backend": "vertex"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            print(f'RESPONSE: {result}')
            
            return result
            
        except Exception as e:
            print(f"Error in Gemini image query: {e}")
            logger.error(f"Error in Gemini image query: {e}")
            # Try text-only fallback for any Gemini error
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name, system_prompt)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e
    
    def get_text_query(self, text: str, module_name: str = "Unknown", system_prompt: Optional[str] = None) -> str:
        """Process a text-only prompt using Gemini API"""
        try:
            start_time = time.time()            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response.")
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."
            
            result = response.text
            
            # Log the interaction
            duration = time.time() - start_time
            log_llm_interaction(
                interaction_type=f"vertex_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "vertex", "has_image": False, "system_prompt": bool(system_prompt)},
                model_info={"model": self.model_name, "backend": "vertex"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            print(f"Error in Gemini text query: {e}")
            logger.error(f"Error in Gemini text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."


class GeminiBackend(VLMBackend):
    """Google Gemini API backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai")
        
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: Gemini API key is missing! Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        self.genai = genai
        
        logger.info(f"Gemini backend initialized with model: {model_name}")

    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff."""
        response = self.model.generate_content(content_parts)
        response.resolve()
        return response
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown", system_prompt: Optional[str] = None) -> str:
        """Process an image and text prompt using Gemini API"""
        start_time = time.time()
        try:
            image = self._prepare_image(img)
            
            # Prepare content for Gemini
            content_parts = [text, image]
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name, system_prompt)
            
            result = response.text
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": True, "token_usage": token_usage, "system_prompt": bool(system_prompt)},
                model_info={"model": self.model_name, "backend": "gemini"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini image query: {e}")
            # Try text-only fallback for any Gemini error
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name, system_prompt)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e
    
    def get_text_query(self, text: str, module_name: str = "Unknown", system_prompt: Optional[str] = None) -> str:
        """Process a text-only prompt using Gemini API"""
        start_time = time.time()
        try:
            # Prepare content for Gemini
            content_parts = self._prepare_content_parts(text, None, system_prompt)
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response.")
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."
            
            result = response.text
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": False, "token_usage": token_usage, "system_prompt": bool(system_prompt)},
                model_info={"model": self.model_name, "backend": "gemini"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."

class VLM:
    """Main VLM class that supports multiple backends"""
    
    BACKENDS = {
        'gemini': GeminiBackend,
        'vertex': VertexBackend,
    }
    
    def __init__(self, model_name: str, backend: str = 'vertex', **kwargs):
        """
        Initialize VLM with specified backend
        
        Args:
            model_name: Name of the model to use
            backend: Backend type ('gemini', 'vertex')
            **kwargs: Additional arguments passed to backend
        """
        self.model_name = model_name
        self.backend_type = backend.lower()
        
        # Auto-detect backend based on model name if not explicitly specified
        
        if self.backend_type not in self.BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend_type}. Available: {list(self.BACKENDS.keys())}")
        
        # Initialize the appropriate backend
        backend_class = self.BACKENDS[self.backend_type]
        
        # Pass port parameter for legacy Ollama backend
        self.backend = backend_class(model_name, **kwargs)
        
        logger.info(f"VLM initialized with {self.backend_type} backend using model: {model_name}")
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown", system_prompt: Optional[str] = None) -> str:
        """Process an image and text prompt, including an optional system prompt."""
        try:
            result = self.backend.get_query(img, text, module_name, system_prompt)
            return result
        except Exception as e:
            duration = 0
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": True, "system_prompt": bool(system_prompt)}
            )
            raise
    
    def get_text_query(self, text: str, module_name: str = "Unknown", system_prompt: Optional[str] = None) -> str:
        """Process a text-only prompt, including an optional system prompt."""
        try:
            result = self.backend.get_text_query(text, module_name, system_prompt)
            return result
        except Exception as e:
            duration = 0
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": False, "system_prompt": bool(system_prompt)}
            )
            raise