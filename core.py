from pathlib import Path
from typing import Final
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ConversationHistory:
    MAX_HISTORY_MESSAGES: Final[int] = 10
    START_HISTORY_MESSAGES: Final[int] = 2
    END_HISTORY_MESSAGES: Final[int] = 6

    def __init__(self, prompt_file: str | Path = "system_prompt.txt") -> None:
        self.prompt_path = Path(prompt_file)
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Файл системної інструкції не знайдено: {self.prompt_path}")
        self.system_prompt: str = self.prompt_path.read_text(encoding="utf-8").strip()
        self.messages: list[dict[str, str]] = []

    def add_request(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_response(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def _trim_history(self) -> None:
        if len(self.messages) > self.MAX_HISTORY_MESSAGES:
            start_part = self.messages[: self.START_HISTORY_MESSAGES]
            end_part = self.messages[-self.END_HISTORY_MESSAGES :]
            self.messages = start_part + end_part

    def get_messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}, *self.messages]


class Bot:
    MODEL_NAME: Final[str] = "Qwen/Qwen3-1.7B"
    MAX_NEW_TOKENS: Final[int] = 300

    def __init__(self, prompt_file: str | Path = "system_prompt.txt") -> None:
        self.history = ConversationHistory(prompt_file=prompt_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype="auto",
        )
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, user_text: str) -> str:
        self.history.add_request(user_text)
        messages = self.history.get_messages()

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = model_inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][input_length:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        self.history.add_response(response_text)
        return response_text
