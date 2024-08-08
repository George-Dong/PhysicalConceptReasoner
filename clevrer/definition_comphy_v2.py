from nscl.datasets.definition import DatasetDefinitionBase
from jacinle.logging import get_logger

logger = get_logger(__file__)


class ComPhyDefinition(DatasetDefinitionBase):
    operation_signatures = [
        # Part 1: CLEVRER dataset.
        ('scene', [], [], 'object_set'),
        ('objects', [], [], 'object_set'),
        ('events', [], [], 'event_set'),
        ('unseen_events', [], [], 'event_set'),
        ('filter', ['concept'], ['object_set'], 'object_set'),
        ('intersect', [], ['object_set', 'object_set'], 'object_set'),
        ('union', [], ['object_set', 'object_set'], 'object_set'),
        
        ('filter_order', ['concept'], ['object_set'], 'object_set'),
        ('negate', [], ['bool'], 'bool'),
        ('belong_to', [], ['object_set', 'object_set'], 'bool'),
        ('filter_status', ['concept'], ['object_set', 'time_set'], 'object_set'),
        ('filter_temporal', ['concept'], ['object_set', 'time_set'], 'object_set'),

        ('query', ['attribute'], ['object'], 'word'),
        ('exist', [], ['object_set'], 'bool'),
        ('count', [], ['object_set'], 'integer'),
        ('get_frame', [], ['object_set', 'time_set'], 'object_set'),
        ('filter_in', [], ['object_set'], 'time_set'),
        ('filter_out', [], ['object_set'], 'time_set'),
        ('filter_before', [], ['time_set'], ['object_set, time_set']),
        ('filter_after', [], ['time_set'], ['object_set, time_set']),
        ('end', [], ['object_set'], 'time_set'),
        ('start', [], ['object_set'], 'time_set'),
        ('filter_collision', [], ['time_set'], ['object_set, time_set']),
        ('get_col_partner', [], ['object', 'col_set'], 'object_set'),
        ('filter_ancestor', [], ['object_set', 'event', 'object_set', 'event'], ['event_mask1', 'event_mask2', 'event_mask3']),
        ('get_counterfact', [], ['object'], 'event_mask1'),
        # Part 2: ComPhy dataset.
        ('filter_light', ['concept'], ['object_set'], 'object_set'),
        ('filter_heavy', ['concept'], ['object_set'], 'object_set'),
        ('filter_charged', ['concept'], ['object_set'], 'object_set'),
        ('filter_uncharged', ['concept'], ['object_set'], 'object_set'),
        ('filter_opposite', ['concept'], ['object_set'], 'object_set'),
        ('query_both', ['attribute'], ['object_set'], 'words'),
        ('filter_mass', [], ['object'], 'mass_value'),
        ('is_lighter', [], ['mass_value', 'mass_value'], 'bool'),
        ('query_direction', ['attribute'], ['object', 'time_set'], 'word'),
        ('filter_same', ['concept'], ['object_set'], 'object_set'),
        ('is_heavier', [], ['mass_value', 'mass_value'], 'bool'),
        ('counterfact_heavier', [], ['object'], 'event_mask1'),
        ('counterfact_lighter', [], ['object'], 'event_mask1'),
        ('counterfact_opposite', [], ['object_set'], 'event_mask1'),
        ('counterfact_uncharged', [], ['object_set'], 'event_mask1'),
    ]

    attribute_concepts = {
        'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
        'material': ['rubber', 'metal'],
        'shape': ['cube', 'sphere', 'cylinder']
    }

    physical_concepts = {
            'mass': ['light', 'heavy'],
            'charge': ['neutral', 'positive', 'negative'],
            }

    relational_concepts = {
        'order': ['first', 'second', 'last'],
        'event1': ['collision']
    }

    temporal_concepts ={
        'status': ['moving', 'stationary'],
        'order': ['first', 'second', 'last'],
        'event2': ['in', 'out' ],
        'time1': ['before', 'after']
            }
    
    time_concepts ={
        'time2': ['start', 'end'],
        }
    
    direction_concepts ={
        'direction': ['left', 'right', 'up', 'down']
        }
