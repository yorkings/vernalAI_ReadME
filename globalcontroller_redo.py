import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union

# --- PHASE IMPORTS ---
from alpha_phase.brainstate import BrainState
from phase_III.rhythm import RhythmEngine
from phase_V.neuromodulator import Neuromodulator
from phase_III.suprise import SurpriseDetector
from phase_VI.episode import WiseEpisodicMemory
from phase_X.sleepmanager import SleepManager
from phase_X.learning import LearningModule 
from phase_XI.motorpolicied import AdaptiveWiseMotor
from phase_XI.wisereflex import ReflexOrchestrator
from phase_XII.intorospection import IntrospectionModule
from phase_XII.identity_core import IdentityCore
from phase_XII.governor import NeuroPlasticGovernor
from phase_XII.valuedriftmonitor import ValueDriftMonitor

# --- SENSORY BRIDGE IMPORTS ---
# Assumes these files are in the same directory or properly installed
from textbridge import VernalTextBridge
from videobridge import VernalVideoBridge
from visionbridge import VernalVisionBridgeClean
from vernalsignal import VernalSignalBridgeClean

class GlobalBrain(nn.Module):
    def __init__(self, config: Dict[str, Any], vocab_size: int):
        super().__init__()
        # 0. UNIVERSAL BLACKBOARD
        self.brain = BrainState()
        self.internal_dim = config.get('latent_dim', 8)

        # 1. THE THALAMUS (Multi-Modal Sensory Hub)
        # We register all 4 bridges. They all contract to the SAME 8D latent space.
        self.bridges = nn.ModuleDict({
            'text': VernalTextBridge(vocab_size, latent_dim=self.internal_dim),
            'video': VernalVideoBridge(latent_dim=self.internal_dim),
            'vision': VernalVisionBridgeClean(latent_dim=self.internal_dim),
            'signal': VernalSignalBridgeClean(latent_dim=self.internal_dim)
        })

        # 2. PERCEPTION & OPTIMIZATION
        self.hierarchy = config['hierarchy'] 
        self.detector = SurpriseDetector(self.brain)
        self.learner = LearningModule(self.brain, self.hierarchy, base_lr=config.get('lr', 0.001))

        # 3. PLANNING & COGNITION
        self.planner = config['planner'] 
        self.rhythm = RhythmEngine(self.brain)
        self.brain.rhythm_engine = self.rhythm

        # 4. IDENTITY & GOVERNANCE
        self.introspection = IntrospectionModule(self.brain, internal_dim=self.internal_dim)
        self.identity = IdentityCore(self.brain, internal_dim=self.internal_dim, goal_dim=config['goal_dim'])
        self.governor = NeuroPlasticGovernor(self.brain)
        self.value_monitor = ValueDriftMonitor(self.brain, dim=config['goal_dim'])
        self.modulator = Neuromodulator(self.brain)

        # 5. MOTOR & SURVIVAL
        self.motor = AdaptiveWiseMotor(self.brain, config['motor_spec'], self.internal_dim)
        self.spinal_cord = ReflexOrchestrator(self.brain, arcs=config.get('reflex_arcs', []))

        # 6. MEMORY & SLEEP
        self.memory = WiseEpisodicMemory(self.brain)
        self.sleep_manager = SleepManager(self.brain, config['replay_learner'], config['replay_scheduler'], self.hierarchy)

    def perceive(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuses multi-modal inputs into a single 8D Thought.
        Logic: If we see Apple AND hear Apple, the vectors reinforce each other.
        """
        combined_latent = torch.zeros((1, self.internal_dim), device=self.brain.device)
        modalities_active = 0

        # A. Process Text (Stateful: Uses active_memory)
        if 'text' in inputs and inputs['text'] is not None:
            # Pass previous hidden state to keep conversation context
            text_8d, self.brain.active_memory = self.bridges['text'].encode(
                inputs['text'], self.brain.active_memory
            )
            combined_latent += text_8d
            modalities_active += 1

        # B. Process Video (Spatio-Temporal)
        if 'video' in inputs and inputs['video'] is not None:
            video_8d = self.bridges['video'].encode(inputs['video'])
            combined_latent += video_8d
            modalities_active += 1

        # C. Process Vision (Static Image)
        if 'vision' in inputs and inputs['vision'] is not None:
            vision_8d = self.bridges['vision'].encode(inputs['vision'])
            combined_latent += vision_8d
            modalities_active += 1
            
        # D. Process Signal (Audio/Sensor)
        if 'signal' in inputs and inputs['signal'] is not None:
            signal_8d = self.bridges['signal'].encode(inputs['signal'])
            combined_latent += signal_8d
            modalities_active += 1

        # Average the thought vector to stay within [-1, 1] range
        if modalities_active > 0:
            combined_latent = combined_latent / modalities_active
        
        return combined_latent

    def express(self, latent_8d: torch.Tensor, modality: str = 'text'):
        """
        Translates the internal 8D thought into the requested reality.
        """
        if modality not in self.bridges:
            raise ValueError(f"Unknown modality: {modality}. Available: {list(self.bridges.keys())}")
        
        return self.bridges[modality].decode(latent_8d)

    def tick(self, 
             inputs: Dict[str, torch.Tensor], 
             target_output_modality: str = 'text',
             reward: float = 0.0):
        """
        The Unified Loop:
        1. Sensation (Multi-modal Fusion) -> 2. Cognition -> 3. Action (Target Modality)
        """
        self.brain.step_time()
        
        # --- PHASE 1: SENSATION & FUSION ---
        # "Apple" (Text) + [Apple Image] -> Strong "Apple" 8D Vector
        s_t_internal = self.perceive(inputs)
        
        # Reflex Check (Using the fused 8D vector)
        reflex_cmd = self.spinal_cord.check_emergencies(s_t_internal)
        if reflex_cmd is not None:
            return self.motor.update_and_smooth(reflex_cmd)

        internal_pre = self.introspection.gather_internal_snapshot()

        # --- PHASE 2: PERCEPTION & ACTION ---
        self.brain.meta.structural_revision = False
        
        # The Hierarchy predicts the NEXT 8D thought
        latent_rep = self.hierarchy.step(s_t_internal)
        
        if self.brain.meta.structural_revision:
            self.learner.refresh_optimizer()
        
        surprise = self.detector.process()
        
        # Decide Action via Planner
        _, action_intent = self.planner.select_action(latent_rep, context_vector=s_t_internal)
        
        # --- PHASE 3: EXPRESSION (The User Request) ---
        # If the user wants a video prediction, we use the Video Bridge Decoder
        final_output = self.express(latent_rep, modality=target_output_modality)
        self.brain.perception.last_action = final_output

        # --- PHASE 4: BIOLOGICAL UPDATES ---
        self.modulator.update() 
        self.rhythm.step()
        self.governor.update_governance(surprise, reward)

        # --- PHASE 5: LEARNING (Self-Supervised) ---
        # We can train on the modality that was present
        # For simplicity, we use the fused 's_t_internal' as the ground truth for now
        # Ideally, you'd predict the next frame/word specifically.
        self.learner.update_from_step(
            s_t=s_t_internal, 
            a_t=action_intent, 
            s_next_real=latent_rep, # Self-supervised: Prediction vs Reality
            reward=reward 
        )
        
        # --- PHASE 6: MAINTENANCE ---
        self_error = self.introspection.learn_self_dynamics(internal_pre, self.introspection.gather_internal_snapshot())
        self.identity.evaluate_stability(self_error)
        
        # Check Fatigue
        self.memory.record_tick()
        if self.brain.meta.fatigue > 0.9:
            self.sleep_manager.run_full_rest_cycle(self.memory)

        return final_output
