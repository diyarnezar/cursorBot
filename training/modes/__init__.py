"""
Training modes package for Project Hyperion
"""

# Import all advanced trainers
from .quick_trainer import QuickTrainer
from .month_trainer import MonthTrainer
from .quarter_trainer import QuarterTrainer
from .half_year_trainer import HalfYearTrainer
from .year_trainer import YearTrainer
from .two_year_trainer import TwoYearTrainer
from .multi_timeframe_trainer import MultiTimeframeTrainer

__all__ = [
    'QuickTrainer',
    'MonthTrainer',
    'QuarterTrainer', 
    'HalfYearTrainer',
    'YearTrainer',
    'TwoYearTrainer',
    'MultiTimeframeTrainer'
] 