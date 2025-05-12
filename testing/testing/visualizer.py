#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Window Visualization for Ad Recommendation System (AdVise Project)

This module provides three separate windows for visualization:
1. Currently detected people and their gaze times
2. Current highest Q-values for different user categories
3. Reinforcement learning status

Authors: [Your Names]
Version: 1.0.0
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
import argparse
import logging
from torch import serialization
import matplotlib
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import threading

matplotlib.use('TkAgg')  # GUI backend setting
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a font that supports various characters
sns.set_style('whitegrid')  # Seaborn style setting

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdVisualization")

# Directory settings
PROJECT_ROOT = "/home/a/A_2025/AdVise-ML/graduate_project"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
USER_DATA_PATH = os.path.join(DATA_DIR, "user_features.json")

# Color schemes
COLORS = {
    'age': sns.color_palette("Blues_d", 7),
    'gender': sns.color_palette("Oranges_d", 2),
    'emotion': sns.color_palette("Greens_d", 3),
    'time': sns.color_palette("Purples_d", 6),
    'weather': sns.color_palette("Reds_d", 4),
    'categories': sns.color_palette("husl", 6)
}

# Ad categories (English only for better display)
AD_CATEGORIES = ['Culture', 'Education', 'Transport', 'Durables', 'Dining', 'Clothing']
SHORT_CATEGORIES = ['Cult', 'Edu', 'Tran', 'Dura', 'Dini', 'Clot']  # For x-axis labels

# English attribute mapping
ATTRIBUTE_MAP = {
    # Age
    "20세미만": "Under 20", 
    "20-30세": "20-30", 
    "31-40세": "31-40", 
    "41-50세": "41-50", 
    "51-60세": "51-60", 
    "61-70세": "61-70", 
    "70세이상": "Over 70",
    # Gender
    "남성": "Male", 
    "여성": "Female",
    # Emotion
    "happy": "Happy",
    "sad": "Sad",
    "neutral": "Neutral",
    # Time
    "06-09시": "6-9 AM", 
    "09-11시": "9-11 AM", 
    "11-15시": "11-3 PM", 
    "15-19시": "3-7 PM", 
    "19-21시": "7-9 PM", 
    "21-06시": "9 PM-6 AM",
    # Weather/Season
    "봄": "Spring", 
    "여름": "Summer", 
    "가을": "Fall", 
    "겨울": "Winter"
}

# Add numpy.core.multiarray.scalar to safe globals for PyTorch 2.6+
try:
    import numpy.core.multiarray
    serialization.add_safe_globals(['scalar'], globs=numpy.core.multiarray.__dict__)
    logger.info("Added numpy.core.multiarray.scalar to safe globals")
except Exception as e:
    logger.warning(f"Could not add scalar to safe globals: {e}")

class BaseVisualizer:
    """Base class for visualizers with common methods"""
    
    def __init__(self, model_path, refresh_interval=1.0):
        """
        Initialize BaseVisualizer
        
        Args:
            model_path (str): Path to DQN model file
            refresh_interval (float): Data refresh interval (seconds)
        """
        self.model_path = model_path
        self.refresh_interval = refresh_interval
        self.last_refresh_time = 0
        
        # Agent data attributes
        self.reward_history = []
        self.loss_history = []
        self.epsilon_history = []
        self.attribute_rewards = {}
        self.q_values_history = {}
        
        # Current detected users data
        self.current_users = []
        self.max_users_to_display = 4
        
        # Common settings
        self.fig = None
        self.ax = None
        self.running = True
        
        # Load initial data
        self._load_data()
        
    def _convert_attribute_names(self, data_dict):
        """Convert Korean attribute names to English"""
        if not data_dict:
            return data_dict
            
        result = {}
        
        for attr_type, attr_data in data_dict.items():
            result[attr_type] = {}
            
            for attr_value, rewards in attr_data.items():
                # Convert attribute value to English if possible
                en_value = ATTRIBUTE_MAP.get(attr_value, attr_value)
                result[attr_type][en_value] = rewards
                
        return result
        
    def _load_data(self):
        """Load visualization data from model file"""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file does not exist: {self.model_path}")
            return False
        
        try:
            # Load PyTorch model
            model_state = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Extract necessary data
            if 'reward_history' in model_state:
                self.reward_history = model_state.get('reward_history', [])
            if 'loss_history' in model_state:
                self.loss_history = model_state.get('loss_history', [])
            if 'epsilon_history' in model_state:
                self.epsilon_history = model_state.get('epsilon_history', [])
            if 'attribute_rewards' in model_state:
                attr_rewards = model_state.get('attribute_rewards', {})
                self.attribute_rewards = self._convert_attribute_names(attr_rewards)
            if 'q_values_history' in model_state:
                self.q_values_history = model_state.get('q_values_history', {})
            
            logger.info(f"Model data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model data: {e}")
            try:
                # Fallback approach
                model_state = torch.load(self.model_path, map_location='cpu')
                
                # Extract data with the same process as above
                if 'reward_history' in model_state:
                    self.reward_history = model_state.get('reward_history', [])
                if 'loss_history' in model_state:
                    self.loss_history = model_state.get('loss_history', [])
                if 'epsilon_history' in model_state:
                    self.epsilon_history = model_state.get('epsilon_history', [])
                if 'attribute_rewards' in model_state:
                    attr_rewards = model_state.get('attribute_rewards', {})
                    self.attribute_rewards = self._convert_attribute_names(attr_rewards)
                if 'q_values_history' in model_state:
                    self.q_values_history = model_state.get('q_values_history', {})
                
                logger.info(f"Model data loaded successfully with fallback method")
                return True
            except Exception as fallback_e:
                logger.error(f"Fallback loading failed: {fallback_e}")
                return False
    
    def _load_current_users(self):
        """Load current detected users data"""
        try:
            if os.path.exists(USER_DATA_PATH):
                with open(USER_DATA_PATH, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                    
                    # Get the most recent entries
                    current_time = time.time()
                    recent_threshold = 10  # Consider entries from the last 10 seconds
                    
                    # Filter recent users
                    recent_users = [
                        user for user in user_data 
                        if 'timestamp' in user and current_time - user['timestamp'] < recent_threshold
                    ]
                    
                    # Sort by gaze time (descending)
                    recent_users.sort(key=lambda x: x.get('gaze_time', 0), reverse=True)
                    
                    # Take top users
                    self.current_users = recent_users[:self.max_users_to_display]
                    
                    # If no recent users, take the latest users regardless of timestamp
                    if not self.current_users and user_data:
                        self.current_users = sorted(
                            user_data[-self.max_users_to_display:],
                            key=lambda x: x.get('gaze_time', 0),
                            reverse=True
                        )
                    
                    logger.info(f"Loaded {len(self.current_users)} current users")
            else:
                # If no user data exists, create dummy data for visualization testing
                self._create_dummy_users()
                
        except Exception as e:
            logger.error(f"Error loading current users: {e}")
            # Create dummy data if error occurs
            self._create_dummy_users()
    
    def _create_dummy_users(self):
        """Create dummy user data for visualization testing"""
        dummy_users = [
            {
                "id": 1,
                "age": "31-40세",
                "gender": "여성",
                "emotion": "happy",
                "time": "11-15시",
                "weather": "봄",
                "gaze_time": 3.2,
                "timestamp": time.time()
            },
            {
                "id": 2,
                "age": "20-30세",
                "gender": "남성",
                "emotion": "neutral",
                "time": "11-15시",
                "weather": "봄",
                "gaze_time": 2.7,
                "timestamp": time.time()
            },
            {
                "id": 3,
                "age": "41-50세",
                "gender": "여성",
                "emotion": "sad",
                "time": "11-15시",
                "weather": "봄",
                "gaze_time": 1.8,
                "timestamp": time.time()
            }
        ]
        self.current_users = dummy_users
        logger.info("Created dummy user data for visualization")
    
    def _get_recommended_category(self, user):
        """Get recommended ad category for a user"""
        # In a real system, this would use the DQN to get the actual recommendation
        # For now, use simple mapping based on attribute rewards
        
        # Create a list of (category, score) pairs
        category_scores = []
        
        # Check each attribute type
        for attr_type in ['age', 'gender', 'emotion', 'time', 'weather']:
            if attr_type not in self.attribute_rewards:
                continue
                
            attr_value = user.get(attr_type, '')
            # Convert to English if needed for lookup
            attr_value_en = ATTRIBUTE_MAP.get(attr_value, attr_value)
            
            if attr_value_en in self.attribute_rewards[attr_type]:
                rewards = self.attribute_rewards[attr_type][attr_value_en]
                if rewards:
                    avg_reward = sum(rewards) / len(rewards)
                    category_scores.append((AD_CATEGORIES[0], avg_reward))  # Default to first category
                    
                    # Try to find best category from Q-values if available
                    if self.q_values_history:
                        for user_type, cat_data in self.q_values_history.items():
                            if attr_value_en in user_type or attr_type in user_type:
                                if cat_data and all(i in cat_data for i in range(len(AD_CATEGORIES))):
                                    q_values = [cat_data[i][-1] if cat_data[i] else 0 for i in range(len(AD_CATEGORIES))]
                                    best_idx = np.argmax(q_values)
                                    category_scores.append((AD_CATEGORIES[best_idx], avg_reward * 1.5))  # Give extra weight
                                break
        
        # If we have scores, return the highest scoring category
        if category_scores:
            best_category = max(category_scores, key=lambda x: x[1])[0]
            return best_category
            
        # Fallback recommendations based on simple rules
        if user.get('emotion') == 'happy' or user.get('emotion') == 'Happy':
            return 'Culture'  # Entertainment for happy people
        elif user.get('age') == '20-30세' or user.get('age') == '20-30':
            return 'Dining'   # Dining for young adults
        else:
            return AD_CATEGORIES[0]  # Default to first category
    
    def _get_user_color(self, user):
        """Get color for user bar based on user attributes"""
        if 'emotion' in user:
            if user['emotion'] == 'happy' or user['emotion'] == 'Happy':
                return 'green'
            elif user['emotion'] == 'sad' or user['emotion'] == 'Sad':
                return 'blue'
            elif user['emotion'] == 'neutral' or user['emotion'] == 'Neutral':
                return 'gray'
        
        # Fallback to gender-based coloring
        if 'gender' in user:
            if user['gender'] == '남성' or user['gender'] == 'Male':
                return 'skyblue'
            elif user['gender'] == '여성' or user['gender'] == 'Female':
                return 'lightpink'
        
        return 'lightgray'  # Default color
    
    def _get_highest_q_values(self):
        """Get highest Q-values for different user categories"""
        highest_q_values = []
        
        # For each main attribute type
        for attr_type in ['age', 'gender', 'emotion']:
            if attr_type not in self.attribute_rewards:
                continue
                
            attr_data = self.attribute_rewards[attr_type]
            if not attr_data:
                continue
                
            # Find highest reward attribute for this type
            best_attr = None
            best_reward = -1
            
            for attr_value, rewards in attr_data.items():
                if not rewards:
                    continue
                avg_reward = sum(rewards) / len(rewards)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_attr = attr_value
            
            if best_attr and best_reward > 0:
                # Find the best category for this attribute
                best_category_idx = 0
                if self.q_values_history:
                    # Try to find actual Q-values
                    for user_type, cat_data in self.q_values_history.items():
                        if best_attr in user_type or attr_type in user_type:
                            # Get the latest Q-values
                            q_values = [cat_data[i][-1] if cat_data[i] else 0 for i in range(len(AD_CATEGORIES))]
                            best_category_idx = np.argmax(q_values)
                            break
                
                highest_q_values.append({
                    'attribute_type': attr_type,
                    'attribute_value': best_attr,
                    'best_category': AD_CATEGORIES[best_category_idx],
                    'reward': best_reward,
                    'q_values': [cat_data[i][-1] if i in cat_data and cat_data[i] else 0 
                                 for i in range(len(AD_CATEGORIES))]
                                if self.q_values_history and user_type in self.q_values_history else None
                })
        
        return highest_q_values

    def _get_learning_status(self):
        """Get current learning status"""
        status = {
            'loss_current': self.loss_history[-1] if self.loss_history else 0,
            'loss_start': self.loss_history[0] if self.loss_history else 0,
            'epsilon_current': self.epsilon_history[-1] if self.epsilon_history else 1.0,
            'reward_avg_recent': np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0,
            'reward_avg_start': np.mean(self.reward_history[:10]) if len(self.reward_history) >= 10 else 0,
            'learning_steps': len(self.loss_history)
        }
        
        # Calculate improvement percentages
        if status['loss_start'] > 0:
            status['loss_reduction'] = ((status['loss_start'] - status['loss_current']) / status['loss_start']) * 100
        else:
            status['loss_reduction'] = 0
            
        if status['reward_avg_start'] > 0:
            status['reward_improvement'] = ((status['reward_avg_recent'] - status['reward_avg_start']) / 
                                           status['reward_avg_start']) * 100
        else:
            status['reward_improvement'] = 0
            
        return status
    
    def update(self):
        """Update method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement update()")
    
    def run(self):
        """Run visualization loop"""
        try:
            # Initial update
            self.update()
            
            while self.running:
                # Update data and graphs
                self.update()
                
                # Small delay
                plt.pause(0.1)
                
                # Check if Figure is closed
                if not plt.fignum_exists(self.fig.number):
                    logger.info("Visualization window closed. Exiting.")
                    self.running = False
                    break
                    
        except KeyboardInterrupt:
            logger.info("Visualization interrupted by keyboard.")
            self.running = False
            
        except Exception as e:
            logger.error(f"Error during visualization: {e}", exc_info=True)
            self.running = False
            
        finally:
            if plt.fignum_exists(self.fig.number):
                plt.close(self.fig)
            logger.info("Visualization terminated.")
    
    def stop(self):
        """Stop the visualization"""
        self.running = False


class UserVisualizer(BaseVisualizer):
    """Visualizer for currently detected users"""
    
    def __init__(self, model_path, refresh_interval=1.0):
        """Initialize user visualizer"""
        super().__init__(model_path, refresh_interval)
        
        # Create figure
        self.fig = plt.figure(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title('AdVise: Currently Detected Users')
        self.ax = self.fig.add_subplot(111)
        
        # Show initial data
        self._load_current_users()
        self._update_plot()
        
        plt.tight_layout()
        plt.show(block=False)
    
    def _update_plot(self):
        """Update user visualization"""
        # Clear previous plot
        self.ax.clear()
        self.ax.set_title('Currently Detected Users', fontsize=14)
        
        if not self.current_users:
            self.ax.text(0.5, 0.5, 'No users currently detected', 
                         ha='center', va='center', 
                         transform=self.ax.transAxes,
                         fontsize=12)
            return
        
        # Set up the plot
        num_users = len(self.current_users)
        bar_width = 0.6
        x = np.arange(num_users)
        
        # Create bars for gaze times
        bars = self.ax.bar(x, 
                           [user.get('gaze_time', 0) for user in self.current_users],
                           width=bar_width,
                           color=[self._get_user_color(user) for user in self.current_users])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{height:.1f}s',
                         ha='center', va='bottom', fontsize=10)
        
        # Add user information below each bar
        for i, user in enumerate(self.current_users):
            # Get English attribute values
            age = ATTRIBUTE_MAP.get(user.get('age', ''), user.get('age', ''))
            gender = ATTRIBUTE_MAP.get(user.get('gender', ''), user.get('gender', ''))
            emotion = ATTRIBUTE_MAP.get(user.get('emotion', ''), user.get('emotion', ''))
            
            user_text = f"ID: {user.get('id', '?')}\n{age}, {gender}\n{emotion}"
            self.ax.text(i, -0.5, user_text, ha='center', va='top', fontsize=10)
            
            # Try to get recommended ad category for this user
            recommended_category = self._get_recommended_category(user)
            if recommended_category:
                self.ax.text(i, -1.5, f"➜ {recommended_category}", 
                             ha='center', va='top', fontsize=10, 
                             fontweight='bold', color='darkgreen')
        
        # Set axis labels and limits
        self.ax.set_ylabel('Gaze Time (seconds)', fontsize=12)
        self.ax.set_xticks([])  # Hide x ticks since we have custom labels
        self.ax.set_ylim(bottom=-2)  # Make room for text labels
        
        # Add timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        self.ax.text(1.0, 1.0, f"Time: {current_time}", 
                     ha='right', va='top', 
                     transform=self.ax.transAxes,
                     fontsize=10, alpha=0.7)
        
        # Add grid for better readability
        self.ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
    
    def update(self):
        """Update visualization"""
        current_time = time.time()
        if current_time - self.last_refresh_time < self.refresh_interval:
            return
            
        # Reload data
        self._load_data()
        self._load_current_users()
        
        # Update visualization
        self._update_plot()
        
        # Refresh figure
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            logger.error(f"Error updating user canvas: {e}")
        
        self.last_refresh_time = current_time


class QValuesVisualizer(BaseVisualizer):
    """Visualizer for Q-values"""
    
    def __init__(self, model_path, refresh_interval=1.0):
        """Initialize Q-values visualizer"""
        super().__init__(model_path, refresh_interval)
        
        # Create figure
        self.fig = plt.figure(figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('AdVise: Q-Values by User Type')
        
        # Show initial data
        self._update_plot()
        
        plt.tight_layout()
        plt.show(block=False)
    
    def _update_plot(self):
        """Update Q-values visualization"""
        # Clear figure
        self.fig.clear()
        
        # Get highest Q-values
        highest_q_values = self._get_highest_q_values()
        
        if not highest_q_values:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No Q-value data available', 
                    ha='center', va='center', 
                    transform=ax.transAxes,
                    fontsize=12)
            return
        
        # Create subplots for each attribute type
        num_types = len(highest_q_values)
        for i, q_data in enumerate(highest_q_values):
            # Create subplot
            ax = self.fig.add_subplot(1, num_types, i+1)
            
            # Plot the Q-values as a bar chart
            bar_data = q_data.get('q_values', [1.0] * len(AD_CATEGORIES))
            if not bar_data:  # Ensure we have data to plot
                bar_data = [1.0] * len(AD_CATEGORIES)
                
            # Normalize to make the highest value 1.0 for better visualization
            max_val = max(bar_data) if max(bar_data) > 0 else 1.0
            normalized_data = [val / max_val for val in bar_data]
            
            # Create bars
            bars = ax.bar(range(len(AD_CATEGORIES)), normalized_data, 
                          color=COLORS['categories'])
            
            # Highlight the highest bar
            best_idx = np.argmax(bar_data)
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('black')
            
            # Add values on top of bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if j == best_idx:
                    # Add star and value for best category
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'★ {bar_data[j]:.2f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
                else:
                    # Only add value for other categories
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{bar_data[j]:.2f}',
                           ha='center', va='bottom', fontsize=7)
            
            # Set title and labels
            attr_type = q_data['attribute_type'].capitalize()
            attr_value = q_data['attribute_value']
            ax.set_title(f"{attr_type}: {attr_value}", fontsize=12)
            
            # Set x-tick labels (shortened category names)
            ax.set_xticks(range(len(SHORT_CATEGORIES)))
            ax.set_xticklabels(SHORT_CATEGORIES, rotation=45, fontsize=8)
            
            # Set y-axis limit
            ax.set_ylim(0, 1.2)
            
            # Add best category text
            best_cat = q_data['best_category']
            ax.text(0.5, -0.15, f"Best: {best_cat}", 
                   transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold')
            
            # Add grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # Add overall title
        self.fig.suptitle('Highest Q-Values by User Type', fontsize=14)
        
        # Add timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        self.fig.text(0.98, 0.02, f"Time: {current_time}", 
                     ha='right', va='bottom', 
                     fontsize=8, alpha=0.7)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for title
    
    def update(self):
        """Update visualization"""
        current_time = time.time()
        if current_time - self.last_refresh_time < self.refresh_interval:
            return
            
        # Reload data
        self._load_data()
        
        # Update visualization
        self._update_plot()
        
        # Refresh figure
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            logger.error(f"Error updating Q-values canvas: {e}")
        
        self.last_refresh_time = current_time


class LearningVisualizer(BaseVisualizer):
    """Visualizer for learning status"""
    
    def __init__(self, model_path, refresh_interval=1.0):
        """Initialize learning status visualizer"""
        super().__init__(model_path, refresh_interval)
        
        # Create figure
        self.fig = plt.figure(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title('AdVise: Reinforcement Learning Status')
        
        # Show initial data
        self._update_plot()
        
        plt.tight_layout()
        plt.show(block=False)
    
    def _create_gauge(self, ax, value, title, text_value, color):
        """Create a gauge chart
        
        Args:
            ax: Matplotlib axis
            value: Value between 0 and 1
            title: Gauge title
            text_value: Text to display in center
            color: Gauge color
        """
        # Constrain value between 0 and 1
        value = max(0, min(1, value))
        
        # Create gauge
        startangle = 90
        endangle = -270
        
        # Background
        ax.pie([1], startangle=startangle, counterclock=False, 
              colors=['lightgray'], wedgeprops={'width': 0.2, 'edgecolor': 'white'})
        
        # Value
        ax.pie([value, 1-value], startangle=startangle, counterclock=False, 
              colors=[color, 'white'], wedgeprops={'width': 0.2, 'edgecolor': 'white'})
        
        # Add text
        ax.text(0, 0, text_value, ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0, -0.5, title, ha='center', va='center', fontsize=12)
        
        # Add min and max labels
        ax.text(-0.9, -0.1, '0%', ha='center', va='center', fontsize=9)
        ax.text(0.9, -0.1, '100%', ha='center', va='center', fontsize=9)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
    
    def _update_plot(self):
        """Update learning status visualization"""
        # Clear figure
        self.fig.clear()
        
        # Get learning status
        status = self._get_learning_status()
        
        if status['learning_steps'] == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No learning data available', 
                    ha='center', va='center', 
                    transform=ax.transAxes,
                    fontsize=12)
            return
        
        # Create 2x2 grid for different status indicators
        gs = GridSpec(2, 2, figure=self.fig, wspace=0.3, hspace=0.3)
        
        # 1. Loss Reduction (top left)
        ax_loss = self.fig.add_subplot(gs[0, 0])
        self._create_gauge(ax_loss, 
                           min(1.0, status['loss_reduction'] / 100), 
                           'Loss Reduction', 
                           f"{status['loss_reduction']:.1f}%",
                           'green')
        
        # 2. Learning Progress (top right)
        ax_progress = self.fig.add_subplot(gs[0, 1])
        self._create_gauge(ax_progress, 
                           1 - status['epsilon_current'], 
                           'Learning Progress', 
                           f"{(1-status['epsilon_current'])*100:.1f}%",
                           'blue')
        
        # 3. Reward Improvement (bottom left)
        ax_reward = self.fig.add_subplot(gs[1, 0])
        self._create_gauge(ax_reward, 
                           min(1.0, max(0, status['reward_improvement']) / 100), 
                           'Reward Improvement', 
                           f"{status['reward_improvement']:.1f}%",
                           'purple')
        
        # 4. Learning Steps (bottom right)
        ax_steps = self.fig.add_subplot(gs[1, 1])
        ax_steps.text(0.5, 0.5, f"{status['learning_steps']}", 
                     ha='center', va='center', fontsize=36, fontweight='bold')
        ax_steps.text(0.5, 0.2, 'Learning Steps', ha='center', va='center', fontsize=14)
        ax_steps.axis('off')
        
        # Add overall title
        self.fig.suptitle('Reinforcement Learning Status', fontsize=14)
        
        # Add timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        self.fig.text(0.98, 0.02, f"Time: {current_time}", 
                     ha='right', va='bottom', 
                     fontsize=8, alpha=0.7)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for title
    
    def update(self):
        """Update visualization"""
        current_time = time.time()
        if current_time - self.last_refresh_time < self.refresh_interval:
            return
            
        # Reload data
        self._load_data()
        
        # Update visualization
        self._update_plot()
        
        # Refresh figure
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            logger.error(f"Error updating learning status canvas: {e}")
        
        self.last_refresh_time = current_time


def run_visualizer(visualizer_class, model_path, refresh_interval):
    """Run a visualizer in a separate thread"""
    visualizer = visualizer_class(model_path, refresh_interval)
    visualizer.run()
    return visualizer


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AdVise Multi-Window Visualization")
    parser.add_argument('--model', type=str, default=os.path.join(MODEL_DIR, "dqn_model.pth"),
                       help='Model file path')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Data refresh interval (seconds)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Change logging level for debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Create and run visualizers in separate threads
    threads = []
    visualizers = []
    
    # User visualizer
    user_thread = threading.Thread(
        target=lambda: visualizers.append(run_visualizer(UserVisualizer, args.model, args.interval))
    )
    threads.append(user_thread)
    
    # Q-values visualizer
    qvalues_thread = threading.Thread(
        target=lambda: visualizers.append(run_visualizer(QValuesVisualizer, args.model, args.interval))
    )
    threads.append(qvalues_thread)
    
    # Learning status visualizer
    learning_thread = threading.Thread(
        target=lambda: visualizers.append(run_visualizer(LearningVisualizer, args.model, args.interval))
    )
    threads.append(learning_thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    try:
        # Wait for all threads to finish (which won't happen unless windows are closed)
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logger.info("Visualization interrupted by keyboard.")
        # Stop all visualizers
        for visualizer in visualizers:
            if visualizer:
                visualizer.stop()
    
    logger.info("All visualizations terminated.")


if __name__ == "__main__":
    main()
    
#흰화면만 뜨고 아무것도 안뜨네
    """
    2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] Gcf.destroy(self)
2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/site-packages/matplotlib/_pylab_helpers.py", line 66, in destroy
2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] manager.destroy()
2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/site-packages/matplotlib/backends/_backend_tk.py", line 608, in destroy
2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] delayed_destroy()
2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/site-packages/matplotlib/backends/_backend_tk.py", line 598, in delayed_destroy
2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] self.window.destroy()
2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/tkinter/__init__.py", line 2341, in destroy
2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] self.tk.call('destroy', self._w)
2025-04-03 23:23:50,672 - AdViseSystem - INFO - [Visualizer] _tkinter.TclError: can't invoke "destroy" command: application has been destroyed
2025-04-03 23:23:50,772 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:23:50,772 - AdVisualization - INFO - Visualization window closed. Exiting program.
2025-04-03 23:23:50,772 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:23:50,772 - AdVisualization - INFO - Visualization terminated.
^C2025-04-03 23:23:52,270 - AdViseSystem - INFO - Program terminated by keyboard interrupt
2025-04-03 23:23:52,270 - AdViseSystem - INFO - AdVise system terminated
(Advise-ML) a@kbsrobo:~/A_2025/AdVise-ML/graduate_project/code$ python v4_advise_system.py --mode=train --simulation --debug
2025-04-03 23:23:59,456 - AdViseSystem - DEBUG - Debug mode enabled
2025-04-03 23:23:59,456 - AdViseSystem - INFO - Starting agent thread
2025-04-03 23:23:59,456 - AdViseSystem - INFO - Agent execution command: /home/a/anaconda3/envs/Advise-ML/bin/python /home/a/A_2025/AdVise-ML/graduate_project/code/dqn_agent.py --mode=train --simulation --interval=1.0 --save_interval=5 --debug
2025-04-03 23:24:00,351 - AdViseSystem - INFO - [Agent] 2025-04-03 23:24:00,351 - AdRecommendation - INFO - CSV 파일 로드 성공
2025-04-03 23:24:00,353 - AdViseSystem - INFO - [Agent] 2025-04-03 23:24:00,353 - AdRecommendation - INFO - CSV 파일 로드 성공
2025-04-03 23:24:00,544 - AdViseSystem - INFO - Agent process terminated (return code: 0)
2025-04-03 23:24:02,459 - AdViseSystem - INFO - Starting visualization thread
2025-04-03 23:24:02,459 - AdViseSystem - INFO - Visualizer execution command: /home/a/anaconda3/envs/Advise-ML/bin/python /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py --model=/home/a/A_2025/AdVise-ML/graduate_project/models/dqn_model.pth --interval=1.0 --debug
2025-04-03 23:24:03,571 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:03,571 - AdVisualization - WARNING - Could not add scalar to safe globals: add_safe_globals() got an unexpected keyword argument 'globs'
2025-04-03 23:24:03,571 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:03,571 - AdVisualization - DEBUG - Debug mode enabled
2025-04-03 23:24:03,576 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:03,575 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:03,576 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:03,575 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:03,576 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:03,576 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:03,576 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:475: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:03,576 - AdViseSystem - INFO - [Visualizer] self.fig = plt.figure(figsize=(10, 6))
2025-04-03 23:24:03,681 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:708: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:03,681 - AdViseSystem - INFO - [Visualizer] self.fig = plt.figure(figsize=(10, 6))
2025-04-03 23:24:03,782 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:583: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:03,782 - AdViseSystem - INFO - [Visualizer] self.fig = plt.figure(figsize=(12, 6))
2025-04-03 23:24:04,149 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:04,146 - AdVisualization - ERROR - Error loading current users: Expecting value: line 769927 column 13 (char 14175968)
2025-04-03 23:24:04,149 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:04,149 - AdVisualization - INFO - Created dummy user data for visualization
2025-04-03 23:24:04,419 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:484: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:04,419 - AdViseSystem - INFO - [Visualizer] plt.show(block=False)
2025-04-03 23:24:04,440 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:715: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:04,440 - AdViseSystem - INFO - [Visualizer] plt.show(block=False)
2025-04-03 23:24:04,460 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:590: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:04,460 - AdViseSystem - INFO - [Visualizer] plt.show(block=False)
2025-04-03 23:24:04,462 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:04,462 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:04,759 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:04,759 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:04,821 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:04,821 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:05,195 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:05,195 - AdVisualization - ERROR - Error loading current users: Expecting value: line 2495413 column 11 (char 46276582)
2025-04-03 23:24:05,195 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:05,195 - AdVisualization - INFO - Created dummy user data for visualization
2025-04-03 23:24:05,358 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:05,349 - AdVisualization - ERROR - Error during visualization: Calling Tcl from different apartment
2025-04-03 23:24:05,358 - AdViseSystem - INFO - [Visualizer] Traceback (most recent call last):
2025-04-03 23:24:05,358 - AdViseSystem - INFO - [Visualizer] File "/home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py", line 441, in run
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/site-packages/matplotlib/pyplot.py", line 758, in pause
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] canvas.start_event_loop(interval)
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/site-packages/matplotlib/backends/_backend_tk.py", line 447, in start_event_loop
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] self._tkcanvas.mainloop()
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/tkinter/__init__.py", line 1458, in mainloop
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] self.tk.mainloop(n)
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] RuntimeError: Calling Tcl from different apartment
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:05,351 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:05,359 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:05,358 - AdVisualization - INFO - Visualization terminated.
2025-04-03 23:24:05,562 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:05,561 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:05,698 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:05,698 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:05,800 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:05,799 - AdVisualization - ERROR - Error during visualization: Calling Tcl from different apartment
2025-04-03 23:24:05,800 - AdViseSystem - INFO - [Visualizer] Traceback (most recent call last):
2025-04-03 23:24:05,800 - AdViseSystem - INFO - [Visualizer] File "/home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py", line 441, in run
2025-04-03 23:24:05,800 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:05,801 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/site-packages/matplotlib/pyplot.py", line 758, in pause
2025-04-03 23:24:05,801 - AdViseSystem - INFO - [Visualizer] canvas.start_event_loop(interval)
2025-04-03 23:24:05,801 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/site-packages/matplotlib/backends/_backend_tk.py", line 447, in start_event_loop
2025-04-03 23:24:05,801 - AdViseSystem - INFO - [Visualizer] self._tkcanvas.mainloop()
2025-04-03 23:24:05,801 - AdViseSystem - INFO - [Visualizer] File "/home/a/anaconda3/envs/Advise-ML/lib/python3.10/tkinter/__init__.py", line 1458, in mainloop
2025-04-03 23:24:05,801 - AdViseSystem - INFO - [Visualizer] self.tk.mainloop(n)
2025-04-03 23:24:05,802 - AdViseSystem - INFO - [Visualizer] RuntimeError: Calling Tcl from different apartment
2025-04-03 23:24:05,802 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:05,799 - AdVisualization - INFO - Visualization terminated.
2025-04-03 23:24:06,632 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:06,632 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:06,767 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:06,767 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:07,673 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:07,673 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:07,803 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:07,803 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:08,707 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:08,707 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:08,838 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:08,839 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:09,748 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:09,748 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:09,878 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:09,878 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:10,787 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:10,787 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:10,920 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:10,920 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:11,823 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:11,823 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:12,006 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:12,006 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:12,916 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:12,915 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:13,053 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:13,053 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:13,961 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:13,961 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:14,099 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:14,099 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:15,003 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:15,003 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:15,127 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:15,128 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:16,036 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:16,035 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:16,175 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:16,176 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:17,082 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:17,082 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:17,210 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:17,210 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:18,116 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:18,116 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:18,243 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:18,243 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:19,147 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:19,147 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:19,322 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:19,322 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:20,228 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:20,227 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:20,362 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:20,362 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:21,274 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:21,274 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:21,413 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:21,413 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:22,317 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:22,317 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:22,446 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:22,447 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:23,351 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:23,351 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:23,477 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:23,478 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:24,386 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:24,386 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:24,520 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:24,520 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:25,428 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:25,428 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:25,558 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:25,558 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:26,462 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:26,462 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:26,640 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:26,640 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:27,546 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:27,546 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:27,681 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:27,681 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:28,590 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:28,589 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:28,719 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:28,719 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:29,623 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:29,623 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:29,751 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:29,751 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:30,656 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:30,656 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:30,785 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:30,785 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:31,694 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:31,694 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:31,835 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:31,836 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:32,741 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:32,741 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:32,925 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:32,925 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:33,828 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:33,828 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:33,953 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:33,953 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
2025-04-03 23:24:34,858 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:34,858 - AdVisualization - INFO - Model data loaded successfully
2025-04-03 23:24:34,986 - AdViseSystem - INFO - [Visualizer] /home/a/A_2025/AdVise-ML/graduate_project/code/visualizer.py:441: UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
2025-04-03 23:24:34,986 - AdViseSystem - INFO - [Visualizer] plt.pause(0.1)
^C2025-04-03 23:24:35,052 - AdViseSystem - INFO - Program terminated by keyboard interrupt
2025-04-03 23:24:35,053 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:35,052 - AdVisualization - INFO - Visualization interrupted by keyboard.
2025-04-03 23:24:35,053 - AdViseSystem - INFO - AdVise system terminated
2025-04-03 23:24:35,053 - AdViseSystem - INFO - [Visualizer] 2025-04-03 23:24:35,053 - AdVisualization - INFO - All visualizations terminated.
    
    
    
    
    
    """