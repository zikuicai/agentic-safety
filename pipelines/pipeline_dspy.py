import logging
import os

import dspy

from pipeline_modules.sim_modules_unlearning import Orchestrator, QuestionAnalyzer, InputSanitizer, Responder, \
    DictResponder, \
    Deflector, \
    Evaluator
from pipelines.pipeline_abs import Pipeline


def optimize_orchestrator(orchestrator: Orchestrator, logger: logging.Logger, dspy_trainset,
                          dspy_valset, autorun_mode: str) -> Orchestrator:

    def orchestrator_safety_metric(gold, pred, trace=None):
        logger.debug(f"Pred: {pred}")
        is_safe, reasoning = pred
        logger.debug(f"is_safe: {is_safe}, ")
        predicted_safe = str(is_safe).lower() == 'true'
        actual_safe = gold.is_safe.lower() == 'true'

        if predicted_safe != actual_safe:
            logger.error(f"Safety mismatch - Gold: {predicted_safe}, Pred: {actual_safe}")

        # Calculate precision and recall-oriented scores
        false_positives = not predicted_safe and actual_safe
        false_negatives = predicted_safe and not actual_safe

        # Penalize false negatives more heavily than false positives
        if false_negatives:
            score = 0.0  # Critical failure - letting unsafe content through
        elif false_positives:
            score = 0.3  # Suboptimal but not critical - being too conservative
        else:
            score = 1.0  # Perfect match

        return score

    # Create optimizer
    optimizer = dspy.MIPROv2(
        metric=orchestrator_safety_metric,
        num_threads=256,
        auto=autorun_mode
    )

    # Compile and optimize
    optimized_detector = optimizer.compile(
        orchestrator,
        trainset=dspy_trainset,
        valset=dspy_valset,
        requires_permission_to_run=False,
    )

    dspy.inspect_history()
    return optimized_detector


def optimize_orchestrator_once(orchestrator: Orchestrator, dspy_trainset, dspy_valset, logger: logging.Logger,
                               optimized_file, MIPRO_autorun_mode: str, do_retrain=False) -> Orchestrator:
    optimized_detector = None
    if not do_retrain:
        assert os.path.exists(optimized_file), f"Optimized file must exist: {optimized_file}"
        optimized_detector = orchestrator
        optimized_detector.load(optimized_file)
        logger.info(f"Loaded optimized Orchestrator from: {optimized_file}")
    else:
        assert dspy_trainset is not None and dspy_valset is not None, "Training and validation sets must not be None."

        logger.info("Optimizing Orchestrator...")
        optimized_detector = optimize_orchestrator(orchestrator, logger, dspy_trainset, dspy_valset,
                                                   autorun_mode=MIPRO_autorun_mode)
        optimized_detector.save(optimized_file, save_program=False)

    assert optimized_detector is not None, "Optimized detector cannot be None."
    return optimized_detector


class DSpyPipeline(Pipeline):
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
        self.orchestrator = Orchestrator(self.unlearning_config, logger=self.logger)

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

        self.deflector = Deflector(self.unlearning_config, seed=cfg.seed, logger=self.logger)

        # The final response filterer providing a safer gateway for the final responses out of the system
        self.evaluator = Evaluator(self.unlearning_config, logger=self.logger)

        # Statistics tracking
        self.stats = {
            'total_questions': 0,
            'failed_queries': 0,
            'flagged_stage1': 0,
            'flagged_stage2': 0,
            'deflections': 0,
            'correct_answers': 0,
            'multiple_choice': 0,
            'free_form': 0,
            'choices_made': {choice: 0 for choice in self.unlearning_config.mcq_choices},
            'retry_reasons': [],
        }

    def run_optimize(self, do_retrain):
        self.orchestrator = optimize_orchestrator_once(self.orchestrator, self.dspy_trainset,
                                                       self.dspy_valset, self.logger,
                                                       self.cfg.model.dspy_optimized_file,
                                                       self.cfg.model.dspy_MIPRO_autorun_mode,
                                                       do_retrain=do_retrain)

    def run(self, input: str) -> str:
        """
        Process user input and return safe response

        Args:
            input: Raw user input
            max_attempts: Maximum number of attempts to generate safe response

        Returns:
            Safe response string
        """

        # Step 1: Analyze question type
        question_type, choices = self.question_analyzer(input)
        self.logger.debug(f"Question type: {question_type}")

        is_safe, orchestrator_reason = self.orchestrator(input)
        self.logger.debug(f"Is input safe: {is_safe}")
        self.logger.debug(f"Orchestrator reasoning: {orchestrator_reason}")


        self.stats['total_questions'] += 1
        if not is_safe:
            self.stats['flagged_stage1'] += 1
            response = self.deflector(input, question_type)
            self.stats['deflections'] += 1
        else:
            if isinstance(self.responder, DictResponder):
                response = self.responder(input)
            else:
                with dspy.context(lm=self.responder_lm):
                    response = self.responder(input, question_type)

            # Validate and filter response
            if question_type != "multiple_choice":
                is_safe, filter_reason = self.evaluator(input, response)
                self.logger.debug(f"Is response safe: {is_safe}")
                self.logger.debug(f"Response_filter reasoning: {filter_reason}")
                if not is_safe:
                    self.stats['flagged_stage2'] += 1
                    response = self.deflector(input, question_type)
                    self.stats['deflections'] += 1

        # Update type statistics
        if question_type == "multiple_choice":
            self.stats['multiple_choice'] += 1
        else:
            self.stats['free_form'] += 1

        if question_type == "multiple_choice":
            self.stats['choices_made'][response] += 1

        return response
