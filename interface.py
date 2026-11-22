from core import Bot


def show() -> None:
    print("=" * 55)
    print("Консольний чатбот на основі Hugging Face (Qwen3-1.7B)")
    print("Введіть запитання або повідомлення для спілкування.")
    print("Для завершення роботи введіть 'вихід' або 'exit'.")
    print("=" * 55)

    try:
        print("Завантаження моделі та ініціалізація бота...")
        bot = Bot()
        print(f"Бот готовий до роботи! Пристрій обчислень: {bot.device}")
    except Exception as e:
        print(f"Помилка ініціалізації [{type(e).__name__}]: {e}")
        return

    while True:
        try:
            user_input = input("Ви: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("Роботу завершено.")
            break

        if not user_input:
            continue

        match user_input.lower():
            case "вихід" | "exit":
                print("До побачення! Завершення роботи чатбота.")
                break
            case _:
                try:
                    response = bot.generate_response(user_input)
                    print(f"Бот: {response}")
                except Exception as e:
                    print(f"Помилка генерації [{type(e).__name__}]: {e}")


if __name__ == "__main__":
    show()
