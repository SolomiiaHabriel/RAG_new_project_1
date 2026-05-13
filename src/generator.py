import ollama
from retrieval import hybrid_search


def build_prompt(question, contexts):
    context_text = "\n\n".join([c["chunk"]["text"] for c in contexts])

    prompt = f"""
Ти — розумна питально-відповідна система.

Відповідай ТІЛЬКИ на основі контексту.
Якщо відповіді немає — скажи:
"У наданому контексті немає достатньо інформації."

Контекст:
{context_text}

Питання:
{question}

Відповідь:
"""
    return prompt


def answer_question(question):
    # 1. Знайти релевантні чанки
    contexts = hybrid_search(question, top_k=5)

    # 2. Побудувати prompt
    prompt = build_prompt(question, contexts)

    # 3. Запит до Mistral через Ollama
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }


if __name__ == "__main__":
    question = "Що таке штучний інтелект?"

    result = answer_question(question)

    print("\nПитання:")
    print(result["question"])

    print("\nВідповідь:")
    print(result["answer"])