#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdVise Project - Gaze-based Ad Recommendation System

This script runs the DQN reinforcement learning-based ad recommendation system and real-time visualization together.
The visualization provides intuitive insight into the learning process and results.

Authors: [Your Names]
Version: 1.0.0
"""

import os
import argparse
import subprocess
import threading
import time
import logging
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advise_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdViseSystem")

# Directory settings
PROJECT_ROOT = "/home/a/A_2025/AdVise-ML/graduate_project"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# File paths for the agent and visualizer
AGENT_SCRIPT = os.path.join(BASE_DIR, "dqn_agent.py")
VISUALIZER_SCRIPT = os.path.join(BASE_DIR, "visualizer.py")

def run_agent(mode, simulation, interval, save_interval, debug):
    """
    Run the DQN agent
    
    Args:
        mode (str): Learning mode ('train' or 'continue')
        simulation (bool): Whether to use simulation mode
        interval (float): Data check interval
        save_interval (int): Model save interval
        debug (bool): Whether to enable debug mode
    """
    cmd = [sys.executable, AGENT_SCRIPT]
    cmd.append(f"--mode={mode}")
    
    if simulation:
        cmd.append("--simulation")
    
    cmd.append(f"--interval={interval}")
    cmd.append(f"--save_interval={save_interval}")
    
    if debug:
        cmd.append("--debug")
    
    logger.info(f"Agent execution command: {' '.join(cmd)}")
    
    # Run agent as subprocess
    try:
        agent_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Log output
        for line in agent_process.stdout:
            logger.info(f"[Agent] {line.strip()}")
            
        # Wait for process to end
        agent_process.wait()
        logger.info(f"Agent process terminated (return code: {agent_process.returncode})")
        
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        
def run_visualizer(model_path, interval, debug):
    """
    Run the visualizer
    
    Args:
        model_path (str): Model file path
        interval (float): Data refresh interval
        debug (bool): Whether to enable debug mode
    """
    cmd = [sys.executable, VISUALIZER_SCRIPT]
    cmd.append(f"--model={model_path}")
    cmd.append(f"--interval={interval}")
    
    if debug:
        cmd.append("--debug")
    
    logger.info(f"Visualizer execution command: {' '.join(cmd)}")
    
    # Run visualizer as subprocess
    try:
        viz_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Log output
        for line in viz_process.stdout:
            logger.info(f"[Visualizer] {line.strip()}")
            
        # Wait for process to end
        viz_process.wait()
        logger.info(f"Visualizer process terminated (return code: {viz_process.returncode})")
        
    except Exception as e:
        logger.error(f"Error running visualizer: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AdVise Ad Recommendation System")
    parser.add_argument('--mode', type=str, choices=['train', 'continue'], default='train',
                       help='Learning mode (train: start new, continue: continue learning)')
    parser.add_argument('--simulation', action='store_true',
                       help='Use simulation mode (use dummy data instead of camera data)')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Data refresh interval (seconds)')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Model save interval (episodes)')
    parser.add_argument('--model', type=str,
                       default=os.path.join(MODEL_DIR, "dqn_model.pth"),
                       help='Model file path')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Change logging level for debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Check and create model directory
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    
    # Run each process in a thread
    agent_thread = threading.Thread(
        target=run_agent,
        args=(args.mode, args.simulation, args.interval, args.save_interval, args.debug)
    )
    
    viz_thread = threading.Thread(
        target=run_visualizer,
        args=(args.model, args.interval, args.debug)
    )
    
    # Start threads
    logger.info("Starting agent thread")
    agent_thread.start()
    
    # Small delay before starting visualization (time for model file creation)
    time.sleep(3)
    
    logger.info("Starting visualization thread")
    viz_thread.start()
    
    try:
        # Wait for all threads to finish
        agent_thread.join()
        viz_thread.join()
    except KeyboardInterrupt:
        logger.info("Program terminated by keyboard interrupt")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
    finally:
        logger.info("AdVise system terminated")

if __name__ == "__main__":
    main()