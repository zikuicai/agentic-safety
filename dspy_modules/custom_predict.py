import logging

from dspy_modules.custom_adapters import CustomAdapter
from dspy.predict.predict import *


def custom_v2_5_generate(lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
    # The adapter decides the behavior of the parser for the model outputs
    out = CustomAdapter()(
        lm, lm_kwargs=lm_kwargs, signature=signature, demos=demos, inputs=inputs, _parse_values=_parse_values
    )
    return out


# v2_5_generate = custom_v2_5_generate


class CustomPredict(Predict):
    """Custom predict class with no parsing of the LM outputs."""

    def __init__(self, signature, _parse_values=True, callbacks=None, **config):
        super().__init__(signature, _parse_values=_parse_values, callbacks=callbacks, **config)

    def forward(self, **kwargs):
        print("CustomPredict.forward...")
        assert not dsp.settings.compiling, "It's no longer ever the case that .compiling is True"

        # Extract the three privileged keyword arguments.
        new_signature = ensure_signature(kwargs.pop("new_signature", None))
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        # Get the right LM to use.
        lm = kwargs.pop("lm", self.lm) or dsp.settings.lm
        assert lm is not None, "No LM is loaded."

        # If temperature is 0.0 but its n > 1, set temperature to 0.7.
        temperature = config.get("temperature")
        temperature = lm.kwargs["temperature"] if temperature is None else temperature
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        if new_signature is not None:
            signature = new_signature
        assert len(signature.output_fields) == 1, 'CustomPredict only supports signatures with one output field.'

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            print(f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}.")

        import dspy
        if isinstance(lm, dspy.LM):
            if not self._parse_values:
                completions = custom_v2_5_generate(lm, config, signature, demos, kwargs,
                                                   _parse_values=self._parse_values)
            else:
                completions = v2_5_generate(lm, config, signature, demos, kwargs, _parse_values=self._parse_values)
        else:
            # logging.error("No LM clients except for `dspy.LM` are supported.")
            raise RuntimeError("No LM clients except for `dspy.LM` are supported.")

        pred = Prediction.from_completions(completions, signature=signature)

        if kwargs.pop("_trace", True) and dsp.settings.trace is not None:
            trace = dsp.settings.trace
            trace.append((self, {**kwargs}, pred))

        return pred
