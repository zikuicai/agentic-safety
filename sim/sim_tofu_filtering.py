import logging
import os

from agents.agent import Agent
from pydantic import BaseModel, Field, constr

from dspy_modules.sim_modules import DictResponder
from sim.sim import Simulator
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


class FiltererOutputSchema(BaseModel):
    choice: constr(pattern=r'^(yes|no)$')


class TofuFilteringSimulator(Simulator):
    def __init__(self, cfg, logger: logging.Logger, unlearning_text: str, responder_lm_conf: dict):
        super().__init__(cfg)

        self.cfg = cfg
        self.logger = logger
        assert self.logger is not None
        self.unlearning_config = cfg.defense
        self.author_names = unlearning_text.split('\n')
        self.author_names = [name.strip() for name in self.author_names if name.strip()]
        logger.info(f"Using author names list: {self.author_names}\n")

        if responder_lm_conf['dict_based']:
            self.responder = DictResponder(response_dict=responder_lm_conf['responses_dict'], logger=self.logger)
        else:
            self.responder = Agent(**responder_lm_conf, sys_prompt="", use_cache=cfg.model.use_cache)


        self.response_filterer = Agent(sys_prompt="", model_provider=cfg.model.model_provider,
                                       model_name=cfg.model.model_name, api_base=cfg.model.api_base,
                                       temperature=cfg.model.temperature, output_schema=FiltererOutputSchema,
                                       use_cache=cfg.model.use_cache)
        self.stats = {
            'total_questions': 0,
            'topic_related': 0,
            'multiple_choice': 0,
            'free_form': 0,
            'choices_made': {choice: 0 for choice in self.unlearning_config.mcq_choices},
            'deflections': 0,
        }

    def run(self, question_text: str) -> str:
        self.stats['total_questions'] += 1

        response = self.responder(question_text)

        filterer_query = (
            f'Does the following contain information about any of these people: {','.join(self.author_names)}? '
            f'Output yes or no. Output one word only.\n\nText: {response}')
        self.logger.info(f"Filterer query: {filterer_query}")
        filterer_response = run_agent_query(self.response_filterer,
                                            query=filterer_query,
                                            use_json=True, possible_outputs={'choice': ['yes', 'no']})
        self.logger.info(f"Filterer response: {filterer_response}")
        if filterer_response['choice'].lower() == 'yes':
            self.stats['topic_related'] += 1
            self.stats['deflections'] += 1
            response = "I'm sorry, I can't answer that question."

        return response
