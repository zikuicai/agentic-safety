import dspy

class TopicDetectorSignature(dspy.Signature):
    # TODO: This is partially messing up with the results for TOFU, but it has been present for our MCQ results. And
    #  the grammatical structure is also incorrect!
    """Determines if input is related to unlearning topic"""
    input_text: str = dspy.InputField()
    is_related: bool = dspy.OutputField()
    reasoning: str = dspy.OutputField()  # Capture the step-by-step reasoning


class GeneratorSignature(dspy.Signature):
    # """Make sure to format your output exactly as instructed without any additional text. Whether the
    # requested format is json or with specific starting and ending delimiters, the format should EXACTLY be
    # followed without any extra leading or trailing text."""

    """Make sure to provide your response exactly in the format that the user requests, without any leading
    or trailing text. The requested output format could be either json (in which case your response MUST
    start and end with curly braces); or with specific starting and ending delimiters such as [[ ## delimiter
    ## ]], in which case you must strictly put the starting and ending delimiters in the beginning and end of
    your responses."""

    input: str = dspy.InputField()
    question_type: str = dspy.InputField()
    response: str = dspy.OutputField()
