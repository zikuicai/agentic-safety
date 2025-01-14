from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter


class CustomAdapter(ChatAdapter):
    def parse(self, signature, completion, _parse_values=False):
        # This adapter is only for the case where we don't want to parse the model outputs.
        assert not _parse_values
        assert len(signature.output_fields) == 1, ('CustomAdapter currently only supports signatures with a single '
                                                   'output field.')
        fields = {list(signature.output_fields.keys())[0]: completion}
        return fields
