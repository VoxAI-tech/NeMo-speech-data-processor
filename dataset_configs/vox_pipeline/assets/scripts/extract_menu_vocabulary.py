#!/usr/bin/env python3
"""Extract and organize menu vocabulary for ASR correction."""

import json
import re
from typing import Dict, List, Set

def extract_menu_vocabulary(menu_file: str) -> Dict[str, List[str]]:
    """Extract menu items and organize by category."""
    
    with open(menu_file, 'r') as f:
        menu_data = json.load(f)
    
    vocabulary = {
        'chicken_items': set(),
        'sauces': set(),
        'sides': set(),
        'drinks': set(),
        'rolls_burgers': set(),
        'sizes': {'small', 'medium', 'large', 'jumbo'},
        'modifiers': {'add', 'no', 'extra', 'with', 'without', 'more', 'less', 'lite'},
        'all_items': set()
    }
    
    for item in menu_data:
        name = item.get('name', '').lower()
        
        # Categorize items
        if any(word in name for word in ['chicken', 'breast', 'leg', 'wings']):
            vocabulary['chicken_items'].add(name)
        elif any(word in name for word in ['sauce', 'garlic', 'chilli', 'mayo', 'tahini', 'bbq']):
            vocabulary['sauces'].add(name)
        elif any(word in name for word in ['chips', 'tabouli', 'coleslaw', 'pickles', 'bread', 'fattoush', 'salad']):
            vocabulary['sides'].add(name)
        elif any(word in name for word in ['pepsi', 'water', 'lipton', 'sunkist', 'mountain dew', '7 up', 'juice', 'red bull']):
            vocabulary['drinks'].add(name)
        elif any(word in name for word in ['roll', 'burger', 'falafel']):
            vocabulary['rolls_burgers'].add(name)
        
        # Add to all items
        vocabulary['all_items'].add(name)
        
        # Also add base menu items without size modifiers
        if any(size in name.lower() for size in ['small', 'medium', 'large', 'jumbo']):
            base_name = name.lower()
            for size in ['small ', 'medium ', 'large ', 'jumbo ']:
                base_name = base_name.replace(size, '')
            base_name = base_name.strip()
            if base_name:
                vocabulary['all_items'].add(base_name)
        
        # Extract option variations
        if 'option_groups' in item:
            for group in item['option_groups']:
                for option in group.get('options', []):
                    option_name = option.get('name', '').lower()
                    if option_name:
                        vocabulary['all_items'].add(option_name)
                        # Categorize options
                        if 'sauce' in option_name:
                            vocabulary['sauces'].add(option_name)
    
    # Convert sets to sorted lists
    result = {}
    for key, value in vocabulary.items():
        if isinstance(value, set):
            result[key] = sorted(list(value))
        else:
            result[key] = value
    
    # Add ONLY obvious misspellings -> correct mappings
    # CONSERVATIVE: Don't force mappings, only fix clear errors
    result['corrections'] = {
        # Clear misspellings of tabouli
        'tubuli': 'tabouli',
        'tabuli': 'tabouli', 
        'tabooly': 'tabouli',
        'tabbouli': 'tabouli',
        'tabbouleh': 'tabouli',  # Alternative spelling to standard
        
        # Clear misspellings of falafel
        'falafal': 'falafel',
        'fallafel': 'falafel',
        'felafel': 'falafel',
        
        # Clear misspellings of hommous  
        'hommus': 'hommous',
        'humus': 'hommous',
        'hummus': 'hommous',
        
        # Coleslaw variations
        'cole slaw': 'coleslaw',
        'colslaw': 'coleslaw',
        
        # DON'T auto-correct these - preserve as spoken:
        # 'garlic': 'garlic sauce',  # Let them say just "garlic"
        # 'chilli': 'chilli sauce',  # Let them say just "chilli"
        # 'quarter chicken': '1/4 chicken',  # Preserve their wording
        # 'half chicken': '1/2 chicken'  # Preserve their wording
    }
    
    return result

if __name__ == "__main__":
    menu_vocab = extract_menu_vocabulary('../menus/el_jannah_menu.json')
    
    # Save to JSON
    with open('../vocabularies/menu_vocabulary.json', 'w') as f:
        json.dump(menu_vocab, f, indent=2)
    
    # Save all unique items as text list
    with open('../vocabularies/menu_items_all.txt', 'w') as f:
        for item in menu_vocab['all_items']:
            f.write(f"{item}\n")
    
    print(f"Extracted {len(menu_vocab['all_items'])} unique menu items")
    print(f"Categories: {list(menu_vocab.keys())}")