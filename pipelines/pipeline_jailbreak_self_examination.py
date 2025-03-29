import logging
import pandas as pd
from dataclasses import dataclass
from agents.agent import Agent
from utils.litellm_utils import get_response_content
from pipelines.pipeline_abs import Pipeline


@dataclass
class ProcessingResult:
    index: int
    forbidden_prompt: str
    prompt: str
    response: str
    success: bool


class JailbreakPipeline(Pipeline):
    def __init__(self, cfg, logger: logging.Logger):
        super().__init__(cfg)
        self.cfg = cfg
        self.logger = logger
        self.lm_conf = {
            "model_name": cfg.model.model_name,
            "api_base": cfg.model.api_base,
            "model_provider": cfg.model.model_provider,
            "use_cache": cfg.model.use_cache,
            "temperature": cfg.model.temperature
        }

        self.lm_agent = Agent(
            sys_prompt="",
            model_provider=self.lm_conf['model_provider'],
            model_name=self.lm_conf['model_name'],
            api_base=self.lm_conf['api_base'],
            temperature=self.lm_conf['temperature'],
            use_cache=self.lm_conf['use_cache'],
            output_schema=None,
        )

    def run(self, idx: int, row: pd.Series, category: str, self_examination_category) -> ProcessingResult:
        try:
            if category == "strong_reject":
                forbidden_prompt = row['forbidden_prompt']
                prompt = row['jailbroken_prompt']
            elif category == "false_refusal":
                forbidden_prompt = None
                prompt = row['prompt']

            response = self.lm_agent(prompt)
            answer = get_response_content(response)


            return ProcessingResult(
                index=idx,
                forbidden_prompt=forbidden_prompt,
                prompt=prompt,
                response=answer,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Error processing row {idx}: {str(e)}")
            return ProcessingResult(
                index=idx,
                forbidden_prompt=forbidden_prompt,
                prompt=row.get('jailbroken_prompt', '') if category == "strong_reject" else row.get('prompt', ''),
                response='',
                success=False
            )
