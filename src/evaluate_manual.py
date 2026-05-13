from generator import answer_question
import pandas as pd


def simple_evaluate():
    df = pd.read_csv("questions/synthetic_questions.csv")

    faithfulness_scores = []
    relevancy_scores = []

    for _, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]

        result = answer_question(question)

        answer = result["answer"]
        contexts = [c["chunk"]["text"] for c in result["contexts"]]

        print("\n====================")
        print("Питання:", question)
        print("Відповідь:", answer)

        # 🔹 Faithfulness (простий варіант)
        faithfulness = any(ans_part in " ".join(contexts) for ans_part in answer.split())
        faithfulness_scores.append(int(faithfulness))

        # 🔹 Relevancy (простий варіант)
        relevancy = any(word in answer.lower() for word in question.lower().split())
        relevancy_scores.append(int(relevancy))

    print("\n===== РЕЗУЛЬТАТИ =====")
    print("Faithfulness:", sum(faithfulness_scores) / len(faithfulness_scores))
    print("Answer relevancy:", sum(relevancy_scores) / len(relevancy_scores))


if __name__ == "__main__":
    simple_evaluate()