import os
import logging
from copy import deepcopy
import dspy
from pipelines.pipeline_abs import Pipeline
from pipeline_modules.sim_modules_jailbreak import Orchestrator, QuestionAnalyzer, Responder, Deflector, Evaluator


def optimize_module(module, logger: logging.Logger, safety_trainset, safety_valset,
                    autorun_mode: str) -> dspy.Module:
    # Define shared safety evaluation metric
    def safety_metric(gold, pred, trace=None):
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
        metric=safety_metric,
        num_threads=256,
        auto=autorun_mode
    )

    # Compile and optimize
    logger.info("Optimizing module...")
    optimized_orchestrator = optimizer.compile(
        module,
        trainset=safety_trainset,
        valset=safety_valset,
        requires_permission_to_run=False,
    )

    return optimized_orchestrator


def optimize_module_once(module, safety_trainset, safety_valset, logger: logging.Logger,
                         optimized_file, MIPRO_autorun_mode: str, force_retrain=False) -> dspy.Module:
    optimized_module = deepcopy(module)
    if not force_retrain and os.path.exists(optimized_file):
        optimized_module.load(optimized_file)
        logger.info(f"Using optimized module from: {optimized_file}")
    else:
        assert safety_trainset is not None and safety_valset is not None, "Training and validation sets must not be None."

        logger.info("Optimizing module...")
        optimized_module = optimize_module(optimized_module, logger, safety_trainset, safety_valset,
                                           autorun_mode=MIPRO_autorun_mode)
        optimized_module.save(optimized_file, save_program=False)

    assert optimized_module is not None, "Optimized module cannot be None."
    return optimized_module


class DSpyPipeline(Pipeline):
    """Main system that orchestrates all components"""

    def __init__(self, cfg, logger: logging.Logger, use_separate_responder_lm=False,
                 use_non_parsing_generator=False,
                 responder_lm_conf=None):
        super().__init__(cfg)

        self.use_separate_responder_lm = use_separate_responder_lm
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

        self.responder_lm = self.lm
        if self.use_separate_responder_lm:
            self.responder_lm = dspy.LM(**self.responder_lm_conf)

        self.defense_config = cfg.defense
        self.dspy_trainset_orchestrator, self.dspy_valset_orchestrator = None, None
        self.dspy_trainset_evaluator, self.dspy_valset_evaluator = None, None

        # Detects whether the input is related to the unlearning topics to be randomly responded to or not
        self.orchestrator = Orchestrator(self.defense_config, logger=self.logger)

        # Determines the type of the question (multiple choice, etc.)
        self.question_analyzer = QuestionAnalyzer(logger=self.logger)

        # The responder that provides legit answers to both multiple choice and free-form questions
        self.responder = Responder(self.defense_config, logger=self.logger,
                                   use_non_parsing_generator=self.use_non_parsing_generator)

        self.deflector = Deflector(self.defense_config, seed=cfg.seed, logger=self.logger)

        # The final evaluator (response filter) providing a safer gateway for the final responses out of the system
        self.evaluator = Evaluator(self.defense_config, logger=self.logger)

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
            'choices_made': {choice: 0 for choice in self.cfg.defense.mcq_choices},
            'retry_reasons': [],
        }

        if cfg.enable_dspy_optimization:
            self.run_optimize(force_retrain=False)

    def set_orchestrator_dspy_datasets(self, ds):
        self.dspy_trainset_orchestrator, self.dspy_valset_orchestrator = ds

    def set_evaluator_dspy_datasets(self, ds):
        self.dspy_trainset_evaluator, self.dspy_valset_evaluator = ds

    def run(self, input: str):
        # output response, and is_deflected
        self.stats['total_questions'] += 1

        # Initial analysis remains the same
        question_type, choices = self.question_analyzer(input)
        self.stats[question_type] = self.stats.get(question_type, 0) + 1

        # Initial orchestration
        is_safe, orchestrator_reason = self.orchestrator(input)
        # self.logger.debug(f"Is input safe: {is_safe}")
        # self.logger.debug(f"Orchestrator reasoning: {orchestrator_reason}")

        if not is_safe:
            response = self.deflector(input, question_type)
            self.stats['flagged_stage1'] += 1
            self.stats['deflections'] += 1
            if question_type == "multiple_choice":
                self.stats['choices_made'][response] += 1
            return response, True

        # Generate response
        with dspy.context(lm=self.responder_lm):
            response = self.responder(input, question_type)

        if question_type == "multiple_choice":
            self.stats['choices_made'][response] += 1
            return response, False

        # Evaluate response
        is_safe, filter_reason = self.evaluator(response)
        # self.logger.debug(f"Is response safe: {is_safe}")
        # self.logger.debug(f"Response_filter reasoning: {filter_reason}")

        if not is_safe:
            self.stats['flagged_stage2'] += 1
            response = self.deflector(input, question_type)
            self.stats['deflections'] += 1
            return response, True

        return response, False

    def run_optimize(self, force_retrain):
        # Orchestrator optimization
        self.logger.info("Running or loading optimization for Orchestrator...")
        self.orchestrator = optimize_module_once(self.orchestrator, self.dspy_trainset_orchestrator,
                                                 self.dspy_valset_orchestrator, self.logger,
                                                 self.cfg.model.dspy_optimized_file_orchestrator,
                                                 self.cfg.model.dspy_MIPRO_autorun_mode,
                                                 force_retrain=force_retrain)

        # Evaluator optimization
        self.logger.info("Running or loading optimization for Evaluator...")
        self.evaluator = optimize_module_once(self.evaluator, self.dspy_trainset_evaluator,
                                              self.dspy_valset_evaluator, self.logger,
                                              self.cfg.model.dspy_optimized_file_evaluator,
                                              self.cfg.model.dspy_MIPRO_autorun_mode,
                                              force_retrain=force_retrain)
