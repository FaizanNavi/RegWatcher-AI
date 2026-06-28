import logging
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

logger = logging.getLogger(__name__)

def evaluate_pipeline(questions: list[str], ground_truths: list[str], answers: list[str], contexts: list[list[str]]):
    logger.info("Starting Ragas evaluation")
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    
    logger.info(f"Ragas Evaluation Complete: {result}")
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Run evaluate_pipeline() with dataset to generate Ragas metrics.")
