import dspy


class GeneratorSignature(dspy.Signature):
    input: str = dspy.InputField()
    question_type: str = dspy.InputField()
    response: str = dspy.OutputField()
