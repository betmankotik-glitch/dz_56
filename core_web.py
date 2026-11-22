from pathlib import Path
from typing import Final
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from web_search import WebSearch


class ConversationHistory:
    MAX_HISTORY_MESSAGES: Final[int] = 10
    START_HISTORY_MESSAGES: Final[int] = 2
    END_HISTORY_MESSAGES: Final[int] = 6

    def __init__(self, prompt_file: str | Path = "system_prompt_web.txt") -> None:
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


class BotWeb:
    MODEL_NAME: Final[str] = "Qwen/Qwen3-1.7B"
    MAX_NEW_TOKENS: Final[int] = 300

    def __init__(
        self,
        prompt_file: str | Path = "system_prompt_web.txt",
        web_prompt_file: str | Path = "web_prompt.txt",
    ) -> None:
        self.history = ConversationHistory(prompt_file=prompt_file)
        web_prompt_path = Path(web_prompt_file)
        if not web_prompt_path.exists():
            raise FileNotFoundError(f"Файл веб-інструкції не знайдено: {web_prompt_path}")
        self.web_prompt: str = web_prompt_path.read_text(encoding="utf-8").strip()

        self.searcher = WebSearch()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype="auto",
        )
        self.model.to(self.device)
        self.model.eval()

    def _build_web_context(self, user_text: str) -> str:
        results = self.searcher.search(user_text)
        if not results:
            return (
                f"{self.web_prompt}"
                f"Початкове запитання користувача: {user_text}"
                "Вебматеріали: результатів пошуку не знайдено."
            )

        materials_blocks: list[str] = []
        for idx, item in enumerate(results, 1):
            url = item["url"]
            title = item["title"]
            snippet = item["snippet"]
            page_content = self.searcher.read_page(url)
            block = (
                f"[Джерело {idx}]"
                f"Назва: {title}"
                f"URL: {url}"
                f"Короткий опис: {snippet}"
                f"Текст сторінки:{page_content if page_content else '(Не вдалося завантажити повний текст сторінки)'}"
            )
            materials_blocks.append(block)

        joined_materials = "---".join(materials_blocks)
        return (
            f"{self.web_prompt}"
            f"Початкове запитання користувача: {user_text}"
            f"Вебматеріали:{joined_materials}"
        )

    def generate_response(self, user_text: str) -> str:
        enriched_request = self._build_web_context(user_text)
        self.history.add_request(enriched_request)
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


if __name__ == "__main__":
    print("Завантаження BotWeb...")
    bot = BotWeb()
    print(f"BotWeb готовий! Пристрій: {bot.device}")
    while True:
        query = input("Ви (Web): ").strip()
        if not query:
            continue
        match query.lower():
            case "вихід" | "exit":
                break
            case _:
                try:
                    print(f"Бот: {bot.generate_response(query)}")
                except Exception as e:
                    print(f"Помилка [{type(e).__name__}]: {e}")
