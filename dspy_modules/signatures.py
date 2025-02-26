import dspy


class TopicDetectorSignature(dspy.Signature):
    """Determines if input is related to unlearning topic"""
    input_text: str = dspy.InputField()
    is_related: bool = dspy.OutputField()
    reasoning: str = dspy.OutputField()  # Capture the step-by-step reasoning


class GeneratorSignature(dspy.Signature):
    input: str = dspy.InputField()
    question_type: str = dspy.InputField()
    response: str = dspy.OutputField()
