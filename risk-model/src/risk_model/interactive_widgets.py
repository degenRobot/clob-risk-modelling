"""Interactive widget configurations for notebooks"""

from ipywidgets import (
    Dropdown, IntSlider, FloatSlider, Button, Output,
    VBox, HBox, Layout, HTML, Label, ToggleButtons,
    interactive, interact_manual
)
from typing import List, Dict, Callable, Optional

# Widget style configurations
WIDGET_STYLES = {
    'description_width': 'initial',
    'layout_width': '400px'
}

BUTTON_STYLES = {
    'fetch': {
        'description': 'Fetch Data',
        'button_style': 'primary',
        'icon': 'download',
        'layout': Layout(width='150px')
    },
    'analyze': {
        'description': 'Run Analysis',
        'button_style': 'success',
        'icon': 'check',
        'layout': Layout(width='150px')
    },
    'stress_test': {
        'description': 'Run Stress Test',
        'button_style': 'danger',
        'icon': 'bolt',
        'layout': Layout(width='150px')
    },
    'export': {
        'description': 'Export Results',
        'button_style': 'info',
        'icon': 'save',
        'layout': Layout(width='150px')
    }
}

def create_market_selector(markets: List[Dict], default: str = 'ETHUSDT') -> Dropdown:
    """Create market selection dropdown"""
    return Dropdown(
        options=[(m['name'], m['binance_symbol_perp']) for m in markets],
        value=default,
        description='Market:',
        style={'description_width': WIDGET_STYLES['description_width']},
        layout=Layout(width=WIDGET_STYLES['layout_width'])
    )

def create_lookback_slider(default: int = 30) -> IntSlider:
    """Create lookback period slider"""
    return IntSlider(
        value=default,
        min=7,
        max=90,
        step=1,
        description='Lookback Days:',
        style={'description_width': WIDGET_STYLES['description_width']},
        layout=Layout(width=WIDGET_STYLES['layout_width'])
    )

def create_risk_parameter_widgets() -> Dict:
    """Create risk parameter adjustment widgets"""
    return {
        'impact': FloatSlider(
            value=1.0,
            min=0.1,
            max=5.0,
            step=0.1,
            description='Max Price Impact %:',
            style={'description_width': WIDGET_STYLES['description_width']},
            layout=Layout(width=WIDGET_STYLES['layout_width'])
        ),
        'leverage': IntSlider(
            value=10,
            min=1,
            max=50,
            step=1,
            description='Max Leverage:',
            style={'description_width': WIDGET_STYLES['description_width']},
            layout=Layout(width=WIDGET_STYLES['layout_width'])
        ),
        'oi_ratio': FloatSlider(
            value=0.25,
            min=0.05,
            max=0.50,
            step=0.05,
            description='OI/Liquidity Ratio:',
            style={'description_width': WIDGET_STYLES['description_width']},
            layout=Layout(width=WIDGET_STYLES['layout_width'])
        ),
        'position_limit': FloatSlider(
            value=0.15,
            min=0.05,
            max=0.30,
            step=0.05,
            description='Position Limit %:',
            style={'description_width': WIDGET_STYLES['description_width']},
            layout=Layout(width=WIDGET_STYLES['layout_width'])
        )
    }

def create_stress_test_selector() -> Dropdown:
    """Create stress test scenario selector"""
    return Dropdown(
        options=[
            ('10% OI Liquidation', 'liquidation_10'),
            ('20% OI Liquidation', 'liquidation_20'),
            ('40% OI Liquidation', 'liquidation_40'),
            ('5% Price Gap', 'gap_5'),
            ('10% Price Gap', 'gap_10'),
            ('20% Price Gap', 'gap_20'),
            ('Cascade Scenario', 'cascade')
        ],
        value='liquidation_20',
        description='Scenario:',
        style={'description_width': WIDGET_STYLES['description_width']},
        layout=Layout(width=WIDGET_STYLES['layout_width'])
    )

def create_chart_type_selector() -> ToggleButtons:
    """Create chart type selector"""
    return ToggleButtons(
        options=[
            ('Depth Chart', 'depth'),
            ('Volatility', 'volatility'),
            ('Volume Profile', 'volume'),
            ('Risk Matrix', 'risk')
        ],
        value='depth',
        description='Chart Type:',
        style={'description_width': WIDGET_STYLES['description_width']}
    )

def create_analysis_layout(widgets_dict: Dict) -> VBox:
    """Create standard analysis layout"""
    return VBox([
        HTML('<h3>Market Analysis Parameters</h3>'),
        HBox([widgets_dict.get('market'), widgets_dict.get('lookback')]),
        HTML('<h3>Risk Parameters</h3>'),
        VBox([
            widgets_dict.get('impact'),
            widgets_dict.get('leverage'),
            widgets_dict.get('oi_ratio'),
            widgets_dict.get('position_limit')
        ]),
        HBox([widgets_dict.get('analyze_button'), widgets_dict.get('export_button')])
    ])

def create_button(button_type: str = 'fetch') -> Button:
    """Create styled button"""
    config = BUTTON_STYLES.get(button_type, BUTTON_STYLES['fetch'])
    return Button(**config)

def create_output_area(height: str = '400px') -> Output:
    """Create output area with styling"""
    return Output(layout=Layout(
        height=height,
        border='1px solid #ddd',
        overflow_y='auto',
        padding='10px'
    ))

def create_interactive_dashboard(markets: List[Dict], 
                               analysis_func: Callable,
                               export_func: Optional[Callable] = None) -> VBox:
    """
    Create complete interactive dashboard
    
    Args:
        markets: List of market configurations
        analysis_func: Function to call for analysis
        export_func: Optional function to call for export
        
    Returns:
        VBox containing the complete dashboard
    """
    # Create widgets
    widgets = {
        'market': create_market_selector(markets),
        'lookback': create_lookback_slider(),
        'analyze_button': create_button('analyze'),
        'export_button': create_button('export'),
        'output': create_output_area()
    }
    
    # Add risk parameters
    risk_widgets = create_risk_parameter_widgets()
    widgets.update(risk_widgets)
    
    # Connect callbacks
    widgets['analyze_button'].on_click(
        lambda b: analysis_func(widgets, widgets['output'])
    )
    
    if export_func:
        widgets['export_button'].on_click(
            lambda b: export_func(widgets, widgets['output'])
        )
    
    # Create layout
    return create_analysis_layout(widgets)

# Pre-configured widget sets
WIDGET_PRESETS = {
    'basic_analysis': {
        'market_selector': True,
        'lookback_slider': True,
        'fetch_button': True
    },
    'risk_analysis': {
        'market_selector': True,
        'lookback_slider': True,
        'risk_parameters': True,
        'analyze_button': True
    },
    'stress_test': {
        'market_selector': True,
        'scenario_selector': True,
        'stress_button': True
    }
}