import logging
import os

import dspy

from dspy_modules.sim_modules import TopicDetector, QuestionAnalyzer, InputSanitizer, Responder, DictResponder, \
    Deflector, \
    ResponseFilter
from sim.sim import Simulator


def optimize_topic_detector(topic_detector: TopicDetector, logger: logging.Logger, dspy_trainset,
                            dspy_valset) -> TopicDetector:
    # Define accuracy metric
    def accuracy_metric(gold, pred, trace=None):
        match = gold.is_related.lower() == str(pred).lower()
        if not match:
            logger.error(f"Prediction mismatch - Gold: {gold}, Pred: {pred}")
        return match

    # Create optimizer
    optimizer = dspy.MIPROv2(
        metric=accuracy_metric,
        num_threads=256,
        auto="heavy"
    )

    # Compile and optimize
    optimized_detector = optimizer.compile(
        topic_detector,
        trainset=dspy_trainset,
        valset=dspy_valset,
        requires_permission_to_run=False,
    )
    dspy.inspect_history()

    return optimized_detector


def optimize_topic_detector_once(topic_detector: TopicDetector, dspy_trainset, dspy_valset, logger: logging.Logger,
                                 optimized_file):
    optimized_detector = topic_detector
    if os.path.exists(optimized_file):
        optimized_detector.load(optimized_file)
        logger.info(f"Using optimized topic detector from: {optimized_file}")
    else:
        assert dspy_trainset is not None and dspy_valset is not None, "Training and validation sets must not be None."

        logger.info("Optimizing topic detector...")
        optimized_detector = optimize_topic_detector(topic_detector, logger, dspy_trainset, dspy_valset)
        optimized_detector.save(optimized_file, save_program=False)

    assert optimized_detector is not None, "Optimized detector cannot be None."
    return optimized_detector


class DSpySimulator(Simulator):
    """Main system that orchestrates all components"""

    def __init__(self, cfg, logger: logging.Logger, dspy_datasets: tuple = None, use_separate_responder_lm=False,
                 use_non_parsing_generator=False,
                 responder_lm_conf=None):
        super().__init__(cfg)

        self.use_separate_responder_lm = use_separate_responder_lm
        self.responder_lm_conf = None
        if self.use_separate_responder_lm:
            assert responder_lm_conf is not None and isinstance(responder_lm_conf, dict)
            self.responder_lm_conf = responder_lm_conf

        self.use_non_parsing_generator = use_non_parsing_generator
        self.cfg = cfg
        self.logger = logger
        assert self.logger is not None
        self.lm = dspy.LM(model=f'{cfg.model.model_provider}/' + cfg.model.model_name, api_base=cfg.model.api_base,
                          provider=cfg.model.model_provider,
                          cache=cfg.model.use_cache, temperature=cfg.model.temperature)
        dspy.configure(lm=self.lm)

        self.dspy_trainset, self.dspy_valset = None, None
        if cfg.enable_dspy_optimization == True and dspy_datasets is not None:
            assert len(dspy_datasets) == 2
            self.dspy_trainset = dspy_datasets[0]
            self.dspy_valset = dspy_datasets[1]

        self.unlearning_config = cfg.defense
        # Detects whether the input is related to the unlearning topics to be randomly responded to or not
        self.topic_detector = TopicDetector(self.unlearning_config, logger=self.logger)

        # Sanitizes the user input from any prompt injection or system behavior override attacks
        self.sanitizer = InputSanitizer(logger=self.logger)

        # Determines the type of the question (multiple choice, etc.)
        self.question_analyzer = QuestionAnalyzer(logger=self.logger)

        self.responder_lm = self.lm
        if self.use_separate_responder_lm:
            if self.responder_lm_conf['dict_based'] == False:
                self.responder_lm = dspy.LM(**self.responder_lm_conf)

        # The responder that provides legit answers to both multiple choice and free-form questions
        if not self.responder_lm_conf or self.responder_lm_conf['dict_based'] == False:
            self.responder = Responder(self.unlearning_config, logger=self.logger,
                                       use_non_parsing_generator=self.use_non_parsing_generator)
        else:
            self.responder = DictResponder(response_dict=self.responder_lm_conf['responses_dict'],
                                           logger=self.logger)

        # The random responder that provides safe responses for unlearning-topic-related queries
        self.deflector = Deflector(self.unlearning_config, seed=cfg.seed, logger=self.logger)

        # The final response filterer providing a safer gateway for the final responses out of the system
        self.response_filter = ResponseFilter(self.unlearning_config, logger=self.logger)

        # Statistics tracking
        self.stats = {
            'total_questions': 0,
            'topic_related': 0,
            'multiple_choice': 0,
            'free_form': 0,
            'choices_made': {choice: 0 for choice in self.unlearning_config.mcq_choices},
            'deflections': 0,
        }

        # Optimize the topic detector
        if cfg.enable_dspy_optimization == True:
            self.topic_detector = optimize_topic_detector_once(self.topic_detector, self.dspy_trainset,
                                                               self.dspy_valset, self.logger,
                                                               self.cfg.model.dspy_optimized_file)

    def run(self, input_text: str) -> str:
        """
        Process user input and return safe response

        Args:
            input_text: Raw user input
            max_attempts: Maximum number of attempts to generate safe response

        Returns:
            Safe response string
        """
        self.stats['total_questions'] += 1

        # Step 1: Analyze question type
        question_type, choices = self.question_analyzer(input_text)
        self.logger.debug(f"Question type: {question_type}")

        # Update type statistics
        if question_type == "multiple_choice":
            self.stats['multiple_choice'] += 1
        else:
            self.stats['free_form'] += 1

        # Step 2: Sanitize input if needed
        if question_type == "multiple_choice":
            sanitized_input = input_text
        else:
            # Temporarily disabled
            sanitized_input = input_text

        self.logger.debug(f"Sanitized input: {sanitized_input}")

        # Step 3: Check if topic-related
        is_topic_related = self.topic_detector(sanitized_input)
        self.logger.debug(f"Is topic related: {is_topic_related}")

        if is_topic_related:
            self.stats['topic_related'] += 1
            response = self.deflector(sanitized_input, question_type)
        else:
            if isinstance(self.responder, DictResponder):
                response = self.responder(sanitized_input)
            else:
                with dspy.context(lm=self.responder_lm):
                    response = self.responder(sanitized_input, question_type)

            # Validate and filter response
            if question_type != "multiple_choice":
                is_safe, reason = self.response_filter(sanitized_input, response)
                if not is_safe:
                    response = self.deflector(sanitized_input, question_type)

        # Update statistics
        if question_type == "multiple_choice":
            self.stats['choices_made'][response] += 1
        elif response == self.unlearning_config.refusal_message:
            self.stats['deflections'] += 1
        return response
