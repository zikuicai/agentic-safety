import os
import logging
import dspy
import asyncio
from dspy_modules.sim_modules_jailbreak import Orchestrator, QuestionAnalyzer, Responder, Deflector, ResponseFilter
from sim.base import Simulator


class AlignedSystemOptimizer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
    def save_optimized_system(self, simulator, save_dir):
        """Save optimized components to files"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subdirectories for components
        orchestrator_path = os.path.join(save_dir, 'orchestrator.json')
        # filter_path = os.path.join(save_dir, 'response_filter.json')
        
        # Save components
        simulator.orchestrator.save(orchestrator_path, save_program=False)
        # simulator.response_filter.save(filter_path, save_program=False)
            
        self.logger.info(f"Saved optimized system to {save_dir}")
        
    def load_optimized_system(self, simulator, save_dir):
        """Load optimized components from files"""
        orchestrator_path = os.path.join(save_dir, 'orchestrator.json')
        # filter_path = os.path.join(save_dir, 'response_filter.json')
        
        if os.path.exists(orchestrator_path):
            simulator.orchestrator.load(orchestrator_path)
            self.logger.info("Loaded optimized orchestrator")
        else:
            self.logger.warning("No saved orchestrator found")
            
        # if os.path.exists(filter_path):
        #     simulator.response_filter.load(filter_path)
        #     self.logger.info("Loaded optimized response filter")
        # else:
        #     self.logger.warning("No saved response filter found")
            
        return simulator

    
    def optimize_system(self, simulator, safety_trainset, safety_valset, save_dir=None):
        """
        Optimize system with aligned safety checking between orchestrator and filter
        using the same safety dataset for both components
        """
        # Check if we have saved optimizations
        if save_dir and os.path.exists(save_dir):
            self.logger.info("Found saved optimizations, loading...")
            return self.load_optimized_system(simulator, save_dir)
        
        # Define shared safety evaluation metric
        def orchestrator_safety_metric(gold, pred, trace=None):
            self.logger.debug(f"Pred: {pred}")
            is_safe, reasoning = pred
            self.logger.debug(f"is_safe: {is_safe}, ")
            predicted_safe = str(is_safe).lower() == 'true'
            actual_safe = gold.is_safe.lower() == 'true'
            
            if predicted_safe != actual_safe:
                self.logger.error(f"Safety mismatch - Gold: {predicted_safe}, Pred: {actual_safe}")
                
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

        # def filter_safety_metric(gold, pred, trace=None):
        #     self.logger.debug(f"Pred: {pred}")
        #     is_safe, reasoning = pred
        #     self.logger.debug(f"is_safe: {is_safe}, ")
        #     predicted_safe = str(is_safe).lower() == 'true'
        #     actual_safe = gold.is_safe.lower() == 'true'
            
        #     if predicted_safe != actual_safe:
        #         self.logger.error(f"Safety mismatch - Gold: {actual_safe}, Pred: {predicted_safe}")
                
        #     # Calculate precision and recall-oriented scores
        #     false_positives = predicted_safe and not actual_safe
        #     false_negatives = not predicted_safe and actual_safe
            
        #     # Penalize false negatives more heavily than false positives
        #     if false_negatives:
        #         score = 0.0  # Critical failure - letting unsafe content through
        #     elif false_positives:
        #         score = 0.3  # Suboptimal but not critical - being too conservative
        #     else:
        #         score = 1.0  # Perfect match
                
        #     return score

        # Optimize orchestrator and filter together
        orchestrator_optimizer = dspy.MIPROv2(
            metric=orchestrator_safety_metric,
            num_threads=24,
            auto="light" # medium, heavy
        )

        # Optimize orchestrator
        self.logger.info("Optimizing orchestrator...")
        simulator.orchestrator = orchestrator_optimizer.compile(
            simulator.orchestrator,
            trainset=safety_trainset,
            valset=safety_valset
        )

        # # Collect filter training data using original safety labels
        # self.logger.info("Collecting filter training data...")
        # filter_trainset, filter_valset = self.create_filter_dataset(
        #     simulator, 
        #     safety_trainset + safety_valset,
        #     save_dir=save_dir
        # )

        # # Optimize filter
        # self.logger.info("Optimizing response filter...")
        # filter_optimizer = dspy.MIPROv2(
        #     metric=filter_safety_metric,
        #     num_threads=24,
        #     auto="light"
        # )
        
        # simulator.response_filter = filter_optimizer.compile(
        #     simulator.response_filter,
        #     trainset=filter_trainset,
        #     valset=filter_valset
        # )

        # Save if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.save_optimized_system(simulator, save_dir)
        return simulator



class DSpySimulator(Simulator):
    """Main system that orchestrates all components"""

    def __init__(self, cfg, logger: logging.Logger, dspy_datasets: tuple = None, use_separate_responder_lm=False,
                 use_non_parsing_generator=False,
                 responder_lm_conf=None):
        super().__init__(cfg)

        self.semaphore = asyncio.Semaphore(5)

        self.use_separate_responder_lm = use_separate_responder_lm
        if self.use_separate_responder_lm:
            assert responder_lm_conf is not None and isinstance(responder_lm_conf, dict)
            self.responder_lm_conf = responder_lm_conf

        self.use_non_parsing_generator = use_non_parsing_generator
        self.cfg = cfg
        self.logger = logger
        assert self.logger is not None
        self.lm = dspy.LM(model=cfg.model.model_name, api_base=cfg.model.api_base, provider=cfg.model.model_provider,
                          cache=cfg.model.use_cache, temperature=cfg.model.temperature)
        dspy.configure(lm=self.lm)

        self.responder_lm = self.lm
        if self.use_separate_responder_lm:
            self.responder_lm = dspy.LM(**self.responder_lm_conf)

        self.unlearning_config = cfg.unlearning
        # Detects whether the input is related to the unlearning topics to be randomly responded to or not
        self.orchestrator = Orchestrator(self.unlearning_config, logger=self.logger)

        # Determines the type of the question (multiple choice, etc.)
        self.question_analyzer = QuestionAnalyzer(logger=self.logger)

        # The responder that provides legit answers to both multiple choice and free-form questions
        self.responder = Responder(self.unlearning_config, logger=self.logger,
                                   use_non_parsing_generator=self.use_non_parsing_generator)

        # The random responder that provides safe responses for unlearning-topic-related queries
        self.deflector = Deflector(self.unlearning_config, seed=cfg.seed, logger=self.logger)

        # The final response filterer providing a safer gateway for the final responses out of the system
        self.response_filter = ResponseFilter(self.unlearning_config, logger=self.logger)

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

        if cfg.enable_dspy_optimization and dspy_datasets:
            assert len(dspy_datasets) == 2
            safety_trainset, safety_valset = dspy_datasets
            save_root = "optimized_prompts/"
            os.makedirs(save_root, exist_ok=True)
            if cfg.run.data.name == 'rapid_response':
                save_dir = save_root + f'rapid_response_' + cfg.model.model_name.split('/')[-1] + f"_v{cfg.model.exp_version}"
            else:
                save_dir = save_root + f'jailbreaking_' + cfg.model.model_name.split('/')[-1] + f"_v{cfg.model.exp_version}"
            optimizer = AlignedSystemOptimizer(cfg.unlearning, logger)
            self = optimizer.optimize_system(
                self, 
                safety_trainset=safety_trainset,
                safety_valset=safety_valset,
                save_dir=save_dir
            )

    async def run(self, input: str) -> str:
        async with self.semaphore:
            return await asyncio.to_thread(self._run_sync, input)
    
    def _run_sync(self, input: str) -> tuple:
        # output response, and is_deflected
        self.stats['total_questions'] += 1
        
        # Initial analysis remains the same
        question_type, choices = self.question_analyzer(input)
        self.stats[question_type] = self.stats.get(question_type, 0) + 1
        
        # Initial orchestration
        is_safe, orchestrator_reason = self.orchestrator(input)
        self.logger.debug(f"Is input safe: {is_safe}")
        self.logger.debug(f"Orchestrator reasoning: {orchestrator_reason}")
        
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
        is_safe, filter_reason = self.response_filter(input, response)
        self.logger.debug(f"Is response safe: {is_safe}")
        self.logger.debug(f"Response_filter reasoning: {filter_reason}")
        
        if not is_safe:
            self.stats['flagged_stage2'] += 1
            response = self.deflector(input, question_type)
            self.stats['deflections'] += 1
            return response, True
        
        return response, False
    