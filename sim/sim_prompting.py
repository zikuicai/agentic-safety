import logging

from agents.agent import Agent
from pydantic import BaseModel, Field, constr

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


class OutputSchema(BaseModel):
    choice: constr(pattern=r'^[A-D]$')
    # choice: str


class RegularSimulator(Simulator):
    def __init__(self, cfg, logger: logging.Logger):
        super().__init__(cfg)

        # self.use_separate_responder_lm = use_separate_responder_lm
        # if self.use_separate_responder_lm:
        #     assert responder_lm_conf is not None and isinstance(responder_lm_conf, dict)
        #     self.responder_lm_conf = responder_lm_conf
        #
        # self.use_non_parsing_generator = use_non_parsing_generator
        self.cfg = cfg
        self.logger = logger
        assert self.logger is not None
        # self.lm = (model=cfg.model.model_name, api_base=cfg.model.api_base, provider=cfg.model.model_provider,
        #                   cache=cfg.model.use_cache, temperature=cfg.model.temperature)
        # dspy.configure(lm=self.lm)

        # self.responder_lm = self.lm
        # if self.use_separate_responder_lm:
        #     self.responder_lm = dspy.LM(**self.responder_lm_conf)
        self.unlearning_config = cfg.defense
        with open(self.unlearning_config.unlearning_text_file, 'r') as f:
            self.unlearning_text = f.read()

        self.prompting_field_name = self.unlearning_config.prompting_field_name
        self.prompting_prefix = self.unlearning_config.prompting_prefix[self.prompting_field_name]
        if '{}' in self.prompting_prefix:
            self.prompting_prefix = self.prompting_prefix.format(self.unlearning_text)

        # Detects whether the input is related to the unlearning topics to be randomly responded to or not
        # self.topic_detector = Agent(system_prompt=self.cfg.topic_detector_sys_prompt,
        #                             model_provider=cfg.model.model_provider,
        #                             model_name=cfg.model.model_name, api_base=cfg.model.api_base,
        #                             temperature=cfg.model.temperature)
        #
        # # Sanitizes the user input from any prompt injection or system behavior override attacks
        # self.sanitizer = Agent(system_prompt="", model_provider=cfg.model.model_provider,
        #                        model_name=cfg.model.model_name, api_base=cfg.model.api_base,
        #                        temperature=cfg.model.temperature)
        #
        # # Determines the type of the question (multiple choice, etc.)
        # self.question_analyzer = Agent(system_prompt="", model_provider=cfg.model.model_provider,
        #                                model_name=cfg.model.model_name, api_base=cfg.model.api_base,
        #                                temperature=cfg.model.temperature)

        # The responder that provides legit answers to both multiple choice and free-form questions
        self.responder = Agent(sys_prompt="", model_provider=cfg.model.model_provider,
                               model_name=cfg.model.model_name, api_base=cfg.model.api_base,
                               temperature=cfg.model.temperature, output_schema=OutputSchema,
                               use_cache=cfg.model.use_cache)

        # The random responder that provides safe responses for unlearning-topic-related queries
        # self.deflector = Agent(system_prompt="", model_provider=cfg.model.model_provider,
        #                        model_name=cfg.model.model_name, api_base=cfg.model.api_base,
        #                        temperature=cfg.model.temperature)
        #
        # # The final response filterer providing a safer gateway for the final responses out of the system
        # self.response_filter = Agent(system_prompt=self.cfg.filterer_sys_prompt,
        #                              model_provider=cfg.model.model_provider,
        #                              model_name=cfg.model.model_name, api_base=cfg.model.api_base,
        #                              temperature=cfg.model.temperature)

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

        # Step 1: Analyze question type
        # question_type, choices = self.question_analyzer(input_text)
        # self.logger.debug(f"Question type: {question_type}")
        # question_type = "multiple_choice"
        # Update type statistics
        question_type = "multiple_choice" if is_mcq else "free_form"
        if question_type == "multiple_choice":
            self.stats['multiple_choice'] += 1
        else:
            self.stats['free_form'] += 1

        # sanitized_input = input_text
        # Step 2: Sanitize input if needed
        # if question_type == "multiple_choice":
        #     # TODO: Why is sanitization not applied in this case?
        #     sanitized_input = input_text
        # else:
        #     # TODO: Temporarily disabled (re-writes text and degrades quality or makes it not pass the constraints
        #     #  passed by the user such as "start every sentence with an A" in MT-bench. sanitized_input =
        #     #  self.sanitizer(input_text)
        #     sanitized_input = input_text

        # self.logger.debug(f"Sanitized input: {sanitized_input}")
        query = f'{self.prompting_prefix}\n\n{question_text}'
        # response = run_agent_query(self.responder,
        #                 query=query,
        #                 use_json=True, possible_outputs={'choice': self.cfg.defense.mcq_choices})

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
