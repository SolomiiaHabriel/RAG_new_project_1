import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

from generator import answer_question


def load_questions():
    df = pd.read_csv("questions/synthetic_questions.csv")
    return df


def run_rag_pipeline(df):
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for _, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]

        result = answer_question(question)

        # витягуємо тексти контексту
        context_texts = [c["chunk"]["text"] for c in result["contexts"]]

        questions.append(question)
        answers.append(result["answer"])
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })


def main():
    print("Завантажую питання...")
    df = load_questions()

    print("Запускаю RAG pipeline...")
    dataset = run_rag_pipeline(df)

    print("Обчислюю метрики RAGAS...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    print("\nРезультати:")
    print(result)


if __name__ == "__main__":
    main()