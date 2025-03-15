import logging

from agents.agent import Agent
from pydantic import BaseModel, Field, constr

from pipelines.pipeline_abs import Pipeline
from utils.litellm_utils import get_response_content


def run_agent_query(agent, query, use_json: bool, possible_outputs=None, **kwargs):
    attempts = 0
    max_tries = 5
    while attempts < max_tries:
        try:
            resp = agent(query, **kwargs)
            resp = get_response_content(resp, to_json=use_json)
            if possible_outputs:
                if use_json:
                    for k in possible_outputs.keys():
                        assert resp[k].lower() in [x.lower() for x in possible_outputs[k]]
                else:
                    assert resp.lower() in [x.lower() for x in possible_outputs]
            return resp
        except Exception as e:
            print(e)
    return None


class OutputSchema(BaseModel):
    choice: constr(pattern=r'^[A-D]$')
    # choice: str


class RegularPipeline(Pipeline):
    def __init__(self, cfg, logger: logging.Logger):
        super().__init__(cfg)

        self.cfg = cfg
        self.logger = logger
        assert self.logger is not None

        self.unlearning_config = cfg.defense
        with open(self.unlearning_config.unsafe_file, 'r') as f:
            self.unlearning_text = f.read()

        self.prompting_field_name = self.unlearning_config.prompting_field_name
        self.prompting_prefix = self.unlearning_config.prompting_prefix[self.prompting_field_name]
        if '{}' in self.prompting_prefix:
            self.prompting_prefix = self.prompting_prefix.format(self.unlearning_text)

        # The responder that provides legit answers to both multiple choice and free-form questions
        self.responder = Agent(sys_prompt="", model_provider=cfg.model.model_provider,
                               model_name=cfg.model.model_name, api_base=cfg.model.api_base,
                               temperature=cfg.model.temperature, output_schema=OutputSchema,
                               use_cache=cfg.model.use_cache)

        # Statistics tracking
        self.stats = {
            'total_questions': 0,
            'topic_related': 0,
            'multiple_choice': 0,
            'free_form': 0,
            'choices_made': {choice: 0 for choice in self.unlearning_config.mcq_choices},
            'deflections': 0,
        }

    def run(self, question_text: str, is_mcq: bool) -> str:
        self.stats['total_questions'] += 1

        # Update type stats
        question_type = "multiple_choice" if is_mcq else "free_form"
        if question_type == "multiple_choice":
            self.stats['multiple_choice'] += 1
        else:
            self.stats['free_form'] += 1

        query = f'{self.prompting_prefix}\n\n{question_text}'
        print(f'Query: {query}')

        choice = None
        # while choice not in self.cfg.defense.mcq_choices:
        #     response = self.responder(query)
        #     response = get_response_content(response, to_json=True)
        #     choice = response['choice']
        if question_type == "multiple_choice":
            response = run_agent_query(self.responder,
                                       query=query,
                                       use_json=True, possible_outputs={'choice': self.cfg.defense.mcq_choices},
                                       use_output_schema=True)
            response = response['choice'][0]
            self.stats['choices_made'][response] += 1
        elif question_type == "free_form":
            response = run_agent_query(self.responder, query=query, use_json=False, use_output_schema=False)

        print(f'Got response: {response}')

        # Update statistics
        # self.stats['deflections'] += 1
        return response
