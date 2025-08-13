"""Command-line interface for Spin-Torque RL-Gym.

Provides easy-to-use CLI commands for training, evaluation, benchmarking,
and deployment of spintronic device control agents.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import yaml

from spin_torque_gym.config import ConfigManager, get_config
from spin_torque_gym.utils.logging_config import setup_logging
from spin_torque_gym.utils.performance import PerformanceProfiler


class SpinTorqueCLI:
    """Main CLI class for Spin-Torque RL-Gym."""

    def __init__(self):
        """Initialize CLI."""
        self.config_manager = None
        self.logger = None

    def setup_environment(self, config_path: Optional[str] = None) -> None:
        """Setup environment and logging."""
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        config = self.config_manager.get_config()
        
        # Setup logging
        self.logger = setup_logging(config.logging)
        
        self.logger.info("Spin-Torque RL-Gym CLI initialized")
        self.logger.info(f"Configuration loaded from: {config_path or 'defaults'}")

    def cmd_info(self, args: argparse.Namespace) -> None:
        """Display system information."""
        config = get_config()
        
        print("🚀 Spin-Torque RL-Gym Information")
        print("=" * 40)
        print(f"Version: 0.1.0")
        print(f"Device Type: {config.device.device_type}")
        print(f"Temperature: {config.physics.temperature}K")
        print(f"Solver: {config.physics.solver_method}")
        print(f"JAX Enabled: {config.compute.use_jax}")
        print(f"GPU Device: {config.compute.gpu_device if config.compute.gpu_device >= 0 else 'CPU'}")
        
        # Environment info
        print("\n📦 Available Environments:")
        environments = ['SpinTorque-v0', 'SpinTorqueArray-v0', 'SkyrmionRacetrack-v0']
        for env_id in environments:
            try:
                env = gym.make(env_id)
                print(f"  ✅ {env_id}")
                print(f"     Action Space: {env.action_space}")
                print(f"     Observation Space: {env.observation_space}")
                env.close()
            except Exception as e:
                print(f"  ❌ {env_id} - Error: {e}")

    def cmd_train(self, args: argparse.Namespace) -> None:
        """Train an RL agent."""
        config = get_config()
        
        print(f"🎯 Training agent for environment: {args.env_id}")
        print(f"Algorithm: {args.algorithm}")
        print(f"Total timesteps: {args.timesteps:,}")
        
        # Create environment
        try:
            env = gym.make(
                args.env_id,
                **self._parse_env_kwargs(args.env_kwargs)
            )
            print(f"✅ Environment created successfully")
        except Exception as e:
            print(f"❌ Failed to create environment: {e}")
            return
        
        # Setup algorithm
        algorithm_class = self._get_algorithm_class(args.algorithm)
        if algorithm_class is None:
            print(f"❌ Unsupported algorithm: {args.algorithm}")
            return
        
        model = algorithm_class(
            'MlpPolicy',
            env,
            verbose=1,
            **self._parse_model_kwargs(args.model_kwargs)
        )
        
        # Training with performance profiling
        profiler = PerformanceProfiler()
        profiler.start_profiling("training")
        
        start_time = time.time()
        try:
            model.learn(total_timesteps=args.timesteps)
            training_time = time.time() - start_time
            
            print(f"✅ Training completed in {training_time:.2f}s")
            
            # Save model
            if args.save_path:
                model.save(args.save_path)
                print(f"💾 Model saved to: {args.save_path}")
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return
        finally:
            profiler.stop_profiling("training")
            
        # Display performance metrics
        stats = profiler.get_stats("training")
        print(f"\n📊 Training Performance:")
        print(f"   Steps/second: {args.timesteps / training_time:.0f}")
        print(f"   Memory usage: {stats.get('peak_memory', 'N/A')}")
        
        env.close()

    def cmd_evaluate(self, args: argparse.Namespace) -> None:
        """Evaluate a trained agent."""
        print(f"🔍 Evaluating model: {args.model_path}")
        print(f"Environment: {args.env_id}")
        print(f"Episodes: {args.episodes}")
        
        # Load model
        try:
            algorithm_class = self._get_algorithm_class(args.algorithm)
            model = algorithm_class.load(args.model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
        
        # Create environment
        try:
            env = gym.make(
                args.env_id,
                render_mode='human' if args.render else None,
                **self._parse_env_kwargs(args.env_kwargs)
            )
        except Exception as e:
            print(f"❌ Failed to create environment: {e}")
            return
        
        # Evaluation
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(args.episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if args.render:
                    env.render()
                
                if done or truncated:
                    if done:
                        success_count += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Reward={episode_reward:.3f}, Length={episode_length}")
        
        # Results summary
        print(f"\n📈 Evaluation Results:")
        print(f"   Success Rate: {success_count/args.episodes:.2%}")
        print(f"   Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"   Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
        # Save results if requested
        if args.save_results:
            results = {
                'success_rate': success_count / args.episodes,
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'mean_length': float(np.mean(episode_lengths)),
                'std_length': float(np.std(episode_lengths)),
                'episode_rewards': [float(r) for r in episode_rewards],
                'episode_lengths': episode_lengths
            }
            
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {args.save_results}")
        
        env.close()

    def cmd_benchmark(self, args: argparse.Namespace) -> None:
        """Run comprehensive benchmarks."""
        print("🏁 Running Spin-Torque RL-Gym Benchmarks")
        print("=" * 50)
        
        # Environment creation benchmark
        print("\n📦 Environment Creation Benchmark:")
        env_ids = ['SpinTorque-v0', 'SpinTorqueArray-v0', 'SkyrmionRacetrack-v0']
        
        for env_id in env_ids:
            try:
                start_time = time.time()
                env = gym.make(env_id)
                creation_time = time.time() - start_time
                env.close()
                print(f"   {env_id}: {creation_time*1000:.2f}ms")
            except Exception as e:
                print(f"   {env_id}: FAILED - {e}")
        
        # Physics simulation benchmark
        print("\n⚛️  Physics Simulation Benchmark:")
        self._benchmark_physics_simulation(args.steps)
        
        # Memory usage benchmark
        print("\n💾 Memory Usage Benchmark:")
        self._benchmark_memory_usage()

    def cmd_config(self, args: argparse.Namespace) -> None:
        """Configuration management."""
        if args.show:
            config = get_config()
            print("⚙️  Current Configuration:")
            print("=" * 30)
            print(yaml.dump(self.config_manager.to_dict(), default_flow_style=False))
            
        elif args.save:
            self.config_manager.save_config(args.save)
            print(f"💾 Configuration saved to: {args.save}")
            
        elif args.validate:
            try:
                self.config_manager._validate_config()
                print("✅ Configuration is valid")
            except ValueError as e:
                print(f"❌ Configuration validation failed: {e}")

    def _get_algorithm_class(self, algorithm: str):
        """Get RL algorithm class."""
        try:
            if algorithm.lower() == 'ppo':
                from stable_baselines3 import PPO
                return PPO
            elif algorithm.lower() == 'sac':
                from stable_baselines3 import SAC
                return SAC
            elif algorithm.lower() == 'td3':
                from stable_baselines3 import TD3
                return TD3
            elif algorithm.lower() == 'dqn':
                from stable_baselines3 import DQN
                return DQN
            else:
                return None
        except ImportError:
            print("❌ stable-baselines3 not found. Install with: pip install stable-baselines3")
            return None

    def _parse_env_kwargs(self, kwargs_str: Optional[str]) -> Dict[str, Any]:
        """Parse environment kwargs from string."""
        if not kwargs_str:
            return {}
        
        try:
            # Simple key=value parsing
            kwargs = {}
            for pair in kwargs_str.split(','):
                key, value = pair.split('=')
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate type
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string
                
                kwargs[key] = value
            
            return kwargs
        except Exception as e:
            print(f"⚠️  Failed to parse environment kwargs: {e}")
            return {}

    def _parse_model_kwargs(self, kwargs_str: Optional[str]) -> Dict[str, Any]:
        """Parse model kwargs from string."""
        return self._parse_env_kwargs(kwargs_str)

    def _benchmark_physics_simulation(self, steps: int) -> None:
        """Benchmark physics simulation performance."""
        try:
            env = gym.make('SpinTorque-v0')
            
            # Warm up
            obs, _ = env.reset()
            for _ in range(10):
                action = env.action_space.sample()
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    obs, _ = env.reset()
            
            # Benchmark
            start_time = time.time()
            obs, _ = env.reset()
            
            for step in range(steps):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                
                if done or truncated:
                    obs, _ = env.reset()
            
            simulation_time = time.time() - start_time
            steps_per_second = steps / simulation_time
            
            print(f"   Physics steps: {steps}")
            print(f"   Total time: {simulation_time:.3f}s")
            print(f"   Steps/second: {steps_per_second:.0f}")
            
            env.close()
            
        except Exception as e:
            print(f"   Physics benchmark failed: {e}")

    def _benchmark_memory_usage(self) -> None:
        """Benchmark memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Create multiple environments
            envs = []
            for _ in range(5):
                envs.append(gym.make('SpinTorque-v0'))
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_per_env = (current_memory - initial_memory) / 5
            
            print(f"   Initial memory: {initial_memory:.1f}MB")
            print(f"   With 5 environments: {current_memory:.1f}MB")
            print(f"   Memory per environment: {memory_per_env:.1f}MB")
            
            # Cleanup
            for env in envs:
                env.close()
                
        except Exception as e:
            print(f"   Memory benchmark failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spin-Torque RL-Gym Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help="Path to configuration file (JSON/YAML)"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an RL agent')
    train_parser.add_argument('--env-id', default='SpinTorque-v0', help='Environment ID')
    train_parser.add_argument('--algorithm', default='PPO', choices=['PPO', 'SAC', 'TD3', 'DQN'])
    train_parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    train_parser.add_argument('--save-path', type=str, help='Path to save trained model')
    train_parser.add_argument('--env-kwargs', type=str, help='Environment kwargs (key=value,key=value)')
    train_parser.add_argument('--model-kwargs', type=str, help='Model kwargs (key=value,key=value)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained agent')
    eval_parser.add_argument('model_path', help='Path to trained model')
    eval_parser.add_argument('--env-id', default='SpinTorque-v0', help='Environment ID')
    eval_parser.add_argument('--algorithm', default='PPO', choices=['PPO', 'SAC', 'TD3', 'DQN'])
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    eval_parser.add_argument('--render', action='store_true', help='Render environment')
    eval_parser.add_argument('--save-results', type=str, help='Save results to JSON file')
    eval_parser.add_argument('--env-kwargs', type=str, help='Environment kwargs')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    bench_parser.add_argument('--steps', type=int, default=1000, help='Physics simulation steps')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--save', type=str, help='Save configuration to file')
    config_parser.add_argument('--validate', action='store_true', help='Validate configuration')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = SpinTorqueCLI()
    cli.setup_environment(args.config)
    
    # Execute command
    try:
        command_method = getattr(cli, f'cmd_{args.command}')
        command_method(args)
    except AttributeError:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()