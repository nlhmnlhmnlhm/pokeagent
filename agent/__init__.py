"""
Agent modules for Pokemon Emerald speedrunning agent
"""

from utils.vlm import VLM
from .simple import SimpleAgent, get_simple_agent, simple_mode_processing_multiprocess, configure_simple_agent_defaults

class Agent:
    """
    Unified agent interface that encapsulates all agent logic.
    The client just calls agent.step(game_state) and gets back an action.
    """
    
    def __init__(self, args=None):
        """
        Initialize the agent based on configuration.
        
        Args:
            args: Command line arguments with agent configuration
        """
        # Extract configuration
        backend = args.backend if args else "gemini"
        model_name = args.model_name if args else "gemini-2.5-flash"
        
        scaffold = args.scaffold if args and hasattr(args, 'scaffold') else "fourmodule"
        
        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name)
        print(f"   VLM: {backend}/{model_name}")
        
        # Initialize agent based on scaffold
        self.scaffold = scaffold
        if scaffold == "simple":
            # Use global SimpleAgent instance to enable checkpoint persistence
            self.agent_impl = get_simple_agent(self.vlm)
            print(f"   Scaffold: Simple (direct frame->action)")

        else:  # fourmodule (default)
            # Four-module agent context
            self.agent_impl = None  # Will use internal four-module processing
            self.context = {
                'perception_output': None,
                'planning_output': None,
                'memory': []
            }
            print(f"   Scaffold: Four-module (Perception->Planning->Memory->Action)")
    
    def step(self, game_state):
        """
        Process a game state and return an action.
        
        Args:
            game_state: Dictionary containing:
                - screenshot: PIL Image
                - game_state: Dict with game memory data
                - visual: Dict with visual observations
                - progress: Dict with milestone progress
        
        Returns:
            dict: Contains 'action' and if a screenshot is ok
        """
        if self.scaffold == "simple":
            # Delegate to specific agent implementation
            if self.scaffold == "simple":
                return self.agent_impl.step(game_state)
                
        if self.scaffold == "fourmodule":
            # Four-module processing
            try:
                # 1. Perception - understand what's happening
                perception_output = perception_step(
                    self.vlm, 
                    game_state, 
                    self.context.get('memory', [])
                )
                self.context['perception_output'] = perception_output
                
                # 2. Planning - decide strategy
                planning_output = planning_step(
                    self.vlm, 
                    perception_output, 
                    self.context.get('memory', [])
                )
                self.context['planning_output'] = planning_output
                
                # 3. Memory - update context
                memory_output = memory_step(
                    perception_output, 
                    planning_output, 
                    self.context.get('memory', [])
                )
                self.context['memory'] = memory_output

                # 4. Feedback - maybe only if stuck ? TODO
                # Patch 2 screenshots to assess if an action is successful TODO : Test if more images if better
                # feedback = screenshot_verif(
                #     self.vlm,
                #     last_screenshot,
                #     game_state["screenshot"],
                #     self.context.get('screenshot_verif', [])
                # )
                # self.context['screenshot_verif'] = feedback
                
                # 5. Action - choose button press
                action_output = action_step(
                    self.vlm, 
                    game_state, 
                    planning_output,
                    perception_output
                )
                
                return action_output
                
            except Exception as e:
                print(f"❌ Agent error: {e}")
                return None

    def report_visual_failure(self, game_state_before, failed_action):
        """
        Passes a visual failure report to the underlying agent implementation.
        
        Args:
            game_state_before: The game state before the failed action.
            failed_action: The action that was reported as failed.
        """
        # Only SimpleAgent implements this
        if self.scaffold == "simple" and hasattr(self.agent_impl, 'report_visual_failure'):
            self.agent_impl.report_visual_failure(game_state_before, failed_action)
        else:
            # 'fourmodule' scaffold doesn't use this, so we can ignore
            pass


__all__ = [
    'Agent',
    'action_step',
    'memory_step',
    'perception_step',
    'planning_step',
    # 'screenshot_verif'
    'SimpleAgent',
    'get_simple_agent',
    'simple_mode_processing_multiprocess',
    'configure_simple_agent_defaults',
]