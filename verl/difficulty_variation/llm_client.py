import os
import re
import time
import json
import base64
import logging
import random
import imghdr
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= Prompts =================
PROMPT_VARIANTS = '''Generate 5 distinct variants of the following problem that:  
1. Preserve the **exact same correct answer** as the original.  
2. Use **significantly different wording, sentence structure**
3. You can adjust the sentence length—either making it concise or extending it—but must ensure correctness.
4. Format of variants as, include <image> in the variants if original has it:  
<variant1>[First variant's full text]</variant1>  
<variant2>[Second variant's full text]</variant2>  
...  
<variant5>[Fifth variant's full text]</variant5>  
**Original Problem:**
'''

PROMPT_THINK_STEP_INIT = r'''You are a mathematician, statistician, and geometer. Below, I will present you with a math problem along with its accompanying diagram (if any).
Given the text, images, generate a step-by-step reasoning process that logically leads to the correct result in the \boxed{}. Requirements:
Flexible step count (3-5 steps): Use only as many steps as needed. Label them clearly with <step1>, <step2>, etc.
Strict dependency on input: Base reasoning only on the provided text and images.
Image reference: Must incorporate details from the image, not just text.
Logical rigor: Ensure each step coherently supports the next.
Please ensure that the thought process is broken down into steps and enclosed within <step></step>.
Format (example with 4 steps):
<step1> [Focus on the observation of the image, describe in detail the information present in the diagram] </step1>
<step2> [Consider what mathematical theorems are available + Logical inference from step1 + text information and image details] </step2>
<step3> [Logical inference continues] </step3>
<step4> [Key conclusion leading to answer] </step4>
\boxed{[final answer]}
'''

# ================= Parsers =================

def add_image_tag_if_missing(text: str) -> str:
    if "<image>" not in text:
        return f"As shown in the figure <image>. {text}"
    return text

def parse_variants(response: str) -> List[str]:
    variants = []
    number = 1
    while True:
        # Try matching multiple formats
        patterns = [
            rf'<variant{number}>(.*?)</variant{number}>',
            rf'<variant{number}>(.*?)(?=<variant{number+1}>)',
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                variants.append(add_image_tag_if_missing(match.group(1).strip()))
                found = True
                break
        if not found:
            break
        number += 1
    return variants

def parse_think_steps(response: str) -> List[str]:
    steps = []
    number = 1
    while True:
        patterns = [
            rf'<step{number}>(.*?)</step{number}>',
            rf'<step{number}>(.*?)(?=<step{number+1}>)',
            rf'<step{number}>(.*?)\boxed',
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                steps.append(match.group(1).strip())
                found = True
                break
        if not found:
            break
        number += 1
    return steps

def parse_boxed_answer(response: str) -> str:
    match = re.search(r'boxed\{(.*?)\}', response) # Simplified match
    if not match:
        match = re.search(r'boxed\s+(.*)', response)
    return match.group(1) if match else ""

# ================= API Base Class =================

class BaseLLMClient(ABC):
    """Base class for API calls"""
    
    @abstractmethod
    def generate_content(self, messages: List[Dict[str, Any]]) -> str:
        pass

    def process_image(self, image_bytes: bytes) -> str:
        """Helper to convert bytes to base64 string with format detection"""
        if not image_bytes:
            return ""
        fmt = imghdr.what(None, h=image_bytes) or 'png'
        # Simple mapping
        fmt_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png'}
        fmt = fmt_map.get(fmt, 'png')
        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/{fmt};base64,{b64_img}"

# ================= Concrete Implementation =================

class AzureQwenClient(BaseLLMClient):
    def __init__(self, api_key: str, endpoint: str, deployment: str = "qwen-vl-max-latest"):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2025-01-01-preview", # Adjust as needed
        )
        self.deployment = deployment

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_content(self, messages: List[Dict[str, Any]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_completion_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

    def generate_variants(self, problem_text: str) -> List[str]:
        """Generate Variants"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT_VARIANTS + problem_text}
        ]
        response = self.generate_content(messages)
        return parse_variants(response)

    def generate_think_steps(self, problem_text: str, correct_answer: str, image_bytes: bytes = None) -> Dict[str, Any]:
        """Generate Think Steps (Extraction + Verification)"""
        
        # 1. Construct Request
        content = [{"type": "text", "text": PROMPT_THINK_STEP_INIT}]
        
        if image_bytes:
            img_url = self.process_image(image_bytes)
            content.append({"type": "image_url", "image_url": {"url": img_url}})
            
        content.append({"type": "text", "text": "Problem: " + problem_text})
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        
        # 2. Generate and Parse
        response = self.generate_content(messages)
        steps = parse_think_steps(response)
        generated_ans = parse_boxed_answer(response)
        
        if not steps:
            return {"status": "failed", "reason": "No steps parsed"}
        
        if not generated_ans:
             return {"status": "failed", "reason": "No boxed answer parsed"}

        # 3. Compare Answers (Check consistency)
        # Performing simple string comparison (stripping whitespace)
        # For more complex mathematical equivalence (e.g., 1/2 vs 0.5), extra libraries would be needed.
        if str(generated_ans).strip() == str(correct_answer).strip():
            return {
                "status": "success",
                "think_steps": steps,
                "think_answer": generated_ans
            }
        else:
            return {
                "status": "failed", 
                "reason": f"Answer mismatch. Generated: '{generated_ans}' vs Dataset: '{correct_answer}'"
            }