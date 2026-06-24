"""
Local LLM adapter for the MAS framework.

Supports:
- mock backend for testing
- transformers backend
- ollama backend

This version is adjusted for:
- Windows UTF-8 subprocess handling
- Ollama CLI stability
- cleaner JSON extraction
- local LLM structured output support
"""

from typing import Optional, Dict, Any
import json
import logging
import re
import subprocess
import shutil
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s'
)


class LocalLLMAgent:
    def __init__(
        self,
        backend: str = 'mock',
        model_name: Optional[str] = None
    ):
        """
        backend:
            one of 'mock', 'transformers', 'ollama'

        model_name:
            local model identifier.
            Example for Ollama:
                llama3:8b
                qwen3:4b
                phi3
        """

        self.backend = backend
        self.model_name = model_name

        logging.info(
            f"Initializing LocalLLMAgent "
            f"with backend={backend}, model_name={model_name}"
        )

        if backend == 'transformers':
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize transformers backend: {e}"
                )

        elif backend == 'ollama':
            try:
                import ollama

                self.ollama = ollama
                self.use_ollama_cli = False

                logging.info(
                    "Using Python 'ollama' client."
                )

            except Exception:
                logging.warning(
                    "Python 'ollama' client not available. "
                    "Will use Ollama CLI via subprocess."
                )

                self.ollama = None
                self.use_ollama_cli = True

                ollama_path = shutil.which("ollama")

                if ollama_path is None:
                    logging.warning(
                        "Ollama CLI was not found in PATH. "
                        "Make sure ollama.exe is installed and available "
                        "from Command Prompt or PowerShell."
                    )
                else:
                    logging.info(
                        f"Ollama CLI found at: {ollama_path}"
                    )

        elif backend == 'mock':
            logging.info(
                "Using mock backend - useful for testing without a model."
            )

        else:
            raise ValueError(
                "Unsupported backend for LocalLLMAgent. "
                "Choose 'mock', 'transformers', or 'ollama'."
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generate a response for the given prompt.

        Returns dictionary:
            {
                'raw': raw_text,
                'tool': parsed tool if available,
                'reason': parsed reason if available,
                'parsed': parsed JSON if available
            }
        """

        if self.backend == 'mock':
            return self._generate_mock(prompt)

        if self.backend == 'transformers':
            return self._generate_transformers(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

        if self.backend == 'ollama':
            return self._generate_ollama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

        raise RuntimeError(
            f"Unsupported backend: {self.backend}"
        )

    # =========================================================
    # MOCK BACKEND
    # =========================================================
    def _generate_mock(
        self,
        prompt: str
    ) -> Dict[str, Any]:

        lower = prompt.lower()

        step_match = re.search(
            r'current step:\s*(\d+)',
            lower
        )

        if step_match:
            step_num = int(step_match.group(1))

            if step_num == 1:
                resp = {
                    'tool': 'load_and_inspect_data',
                    'reason': 'Starting workflow by loading the dataset.',
                    'finish': False
                }

            elif step_num == 2:
                resp = {
                    'tool': 'preprocess_data',
                    'reason': 'Dataset has been loaded and must be preprocessed before analysis.',
                    'finish': False
                }

            elif step_num == 3:
                resp = {
                    'tool': 'analyze_data',
                    'reason': 'Preprocessed data is ready for model analysis.',
                    'finish': False
                }

            elif step_num == 4:
                resp = {
                    'tool': 'generate_recommendations',
                    'reason': 'Model analysis is complete and recommendations are needed.',
                    'finish': False
                }

            else:
                resp = {
                    'tool': None,
                    'reason': 'Workflow complete.',
                    'finish': True
                }

        else:
            resp = {
                'tool': 'load_and_inspect_data',
                'reason': 'Starting workflow by loading data.',
                'finish': False
            }

        return {
            'raw': json.dumps(resp),
            'tool': resp.get('tool'),
            'reason': resp.get('reason'),
            'parsed': resp
        }

    # =========================================================
    # TRANSFORMERS BACKEND
    # =========================================================
    def _generate_transformers(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2
    ) -> Dict[str, Any]:

        input_ids = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).input_ids

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature
        )

        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        parsed = self._extract_json_from_text(text)

        if parsed:
            return {
                'raw': text,
                'tool': parsed.get('tool'),
                'reason': parsed.get('reason'),
                'parsed': parsed
            }

        return {
            'raw': text,
            'tool': None,
            'reason': '',
            'parsed': None
        }

    # =========================================================
    # OLLAMA BACKEND
    # =========================================================
    def _generate_ollama(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2
    ) -> Dict[str, Any]:

        try:
            prompt = self._add_json_instruction(prompt)

            if (
                self.ollama is not None
                and not getattr(self, 'use_ollama_cli', False)
            ):
                text = self._generate_ollama_python_client(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

            else:
                text = self._generate_ollama_cli(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

            text = self._clean_output(text)

            parsed = self._extract_json_from_text(text)

            if parsed:
                return {
                    'raw': text,
                    'tool': parsed.get('tool'),
                    'reason': parsed.get('reason'),
                    'parsed': parsed
                }

            return {
                'raw': text,
                'tool': None,
                'reason': '',
                'parsed': None
            }

        except Exception as e:
            raise RuntimeError(
                f"Ollama generation failed: {e}"
            )

    def _generate_ollama_python_client(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2
    ) -> str:

        options = {
            "temperature": temperature,
            "num_predict": max_tokens
        }

        try:
            response = self.ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options=options,
                format="json"
            )

        except TypeError:
            response = self.ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options=options
            )

        if isinstance(response, dict):
            return (
                response.get('response')
                or response.get('output')
                or response.get('text')
                or str(response)
            )

        return str(response)

    def _generate_ollama_cli(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2
    ) -> str:

        ollama_path = shutil.which("ollama")

        if ollama_path is None:
            raise RuntimeError(
                "Ollama CLI not found in PATH. "
                "Open PowerShell and test: ollama list"
            )

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        command = [
            ollama_path,
            "run",
            self.model_name,
            prompt
        ]

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Ollama CLI failed: {proc.stderr.strip()}"
            )

        return proc.stdout.strip()

    # =========================================================
    # PROMPT / OUTPUT HELPERS
    # =========================================================
    def _add_json_instruction(
        self,
        prompt: str
    ) -> str:

        extra_instruction = """
        
IMPORTANT OUTPUT FORMAT:
Return ONLY one valid JSON object.
Do not include markdown.
Do not include explanation before JSON.
Do not include explanation after JSON.
Do not include thinking text.
Do not include code block fences.

Valid examples:
{"tool":"load_and_inspect_data","reason":"The dataset must be loaded first.","finish":false}
{"tool":"preprocess_data","reason":"The dataset is loaded and must be preprocessed.","finish":false}
{"tool":"analyze_data","reason":"Preprocessed data is ready for model analysis.","finish":false}
{"tool":"generate_recommendations","reason":"Analysis is complete and recommendations are needed.","finish":false}
{"tool":null,"reason":"Workflow completed successfully.","finish":true}
"""

        return prompt + "\n" + extra_instruction

    def _clean_output(
        self,
        text: str
    ) -> str:

        if text is None:
            return ""

        cleaned = str(text).strip()

        cleaned = cleaned.replace("```json", "")
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.replace("`", "")

        # Remove common reasoning tags
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )

        cleaned = cleaned.replace("Thinking...", "")
        cleaned = cleaned.replace("...done thinking.", "")

        return cleaned.strip()

    def _extract_json_from_text(
        self,
        text: str
    ):

        import json
        import re

        if not text:
            return None

        try:
            cleaned = str(text)

            # remove markdown wrappers
            cleaned = cleaned.replace("```json", "")
            cleaned = cleaned.replace("```", "")

            # remove think tags
            cleaned = re.sub(
                r"<think>.*?</think>",
                "",
                cleaned,
                flags=re.DOTALL | re.IGNORECASE
            )

            # ambil JSON object terakhir yang valid
            json_candidates = re.findall(
                r'\{[^{}]*\}',
                cleaned,
                flags=re.DOTALL
            )

            if not json_candidates:
                return None

            # coba parse dari belakang
            for candidate in reversed(json_candidates):

                try:
                    candidate = candidate.strip()

                    parsed = json.loads(candidate)

                    if isinstance(parsed, dict):
                        return parsed

                except Exception:
                    continue

        except Exception as e:
            logging.warning(
                f"JSON extraction failed: {e}"
            )

        return None


if __name__ == '__main__':
    agent = LocalLLMAgent(backend='mock')

    result = agent.generate(
        "Start a workflow to analyze digital marketing campaign data "
        "and recommend advertising decisions."
    )

    print(result)