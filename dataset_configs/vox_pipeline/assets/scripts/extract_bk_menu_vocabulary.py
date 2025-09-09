#!/usr/bin/env python3
"""Extract and organize BK Poland menu vocabulary for ASR correction."""

import json
import re
from typing import Dict, List, Set

def extract_bk_menu_vocabulary(menu_file: str) -> Dict[str, List[str]]:
    """Extract menu items from BK Poland and organize by category."""
    
    with open(menu_file, 'r', encoding='utf-8') as f:
        menu_data = json.load(f)
    
    vocabulary = {
        'burgers': set(),
        'chicken_items': set(),
        'sauces': set(),
        'sides': set(),
        'drinks': set(),
        'desserts': set(),
        'sizes': {'mały', 'średni', 'duży', 'small', 'medium', 'large'},
        'modifiers': {'dodaj', 'bez', 'ekstra', 'więcej', 'mniej', 'dodatkowy', 'dodatkowa', 'dodatkowe'},
        'all_items': set()
    }
    
    for item in menu_data:
        name = item.get('name', '')
        name_lower = name.lower()
        
        # Add original name
        vocabulary['all_items'].add(name)
        vocabulary['all_items'].add(name_lower)
        
        # Categorize items
        if any(word in name_lower for word in ['whopper', 'burger', 'big king', 'steakhouse', 'bacon']):
            vocabulary['burgers'].add(name)
            vocabulary['burgers'].add(name_lower)
        elif any(word in name_lower for word in ['chicken', 'crispy', 'nuggets', 'wings', 'kurczak']):
            vocabulary['chicken_items'].add(name)
            vocabulary['chicken_items'].add(name_lower)
        elif any(word in name_lower for word in ['sos', 'sauce', 'mayo', 'ketchup', 'bbq', 'cheese']):
            vocabulary['sauces'].add(name)
            vocabulary['sauces'].add(name_lower)
        elif any(word in name_lower for word in ['frytki', 'fries', 'salad', 'sałat']):
            vocabulary['sides'].add(name)
            vocabulary['sides'].add(name_lower)
        elif any(word in name_lower for word in ['pepsi', 'cola', 'woda', 'water', 'sok', 'juice']):
            vocabulary['drinks'].add(name)
            vocabulary['drinks'].add(name_lower)
        
        # Extract all options and modifications
        if 'option_groups' in item:
            for group in item['option_groups']:
                for option in group.get('options', []):
                    option_name = option.get('name', '')
                    if option_name:
                        vocabulary['all_items'].add(option_name)
                        vocabulary['all_items'].add(option_name.lower())
                        
                        # Categorize options
                        if any(word in option_name.lower() for word in ['sos', 'sauce', 'mayo', 'ketchup']):
                            vocabulary['sauces'].add(option_name)
                            vocabulary['sauces'].add(option_name.lower())
                        
                        # Extract modifiers like "Extra bacon", "NO cheese", etc.
                        if option_name.startswith(('Extra ', 'NO ', 'Dodatkowy', 'Dodatkowa', 'Dodatkowe', 'Usuń')):
                            vocabulary['all_items'].add(option_name)
    
    # Convert sets to sorted lists
    result = {}
    for key, value in vocabulary.items():
        if isinstance(value, set):
            result[key] = sorted(list(value))
        else:
            result[key] = value
    
    # Polish-specific corrections and common misspellings
    result['corrections'] = {
        # Whopper variations
        'wopper': 'whopper',
        'woper': 'whopper',
        'whooper': 'whopper',
        'wuper': 'whopper',
        
        # Polish words
        'frytky': 'frytki',
        'frytek': 'frytki',
        'serowy': 'cheese',
        'chese': 'cheese',
        
        # Bacon variations
        'bekon': 'bacon',
        'bejkon': 'bacon',
        'beken': 'bacon',
        
        # BBQ variations  
        'barbecue': 'bbq',
        'barbeque': 'bbq',
        
        # Mayo variations
        'majonez': 'mayo',
        'mayonez': 'mayo',
        'mayonnaise': 'mayo',
        
        # Ketchup variations
        'keczup': 'ketchup',
        'ketczup': 'ketchup',
        
        # Pickles
        'pikle': 'pickles',
        'ogórki': 'pickles',
        
        # Common Polish menu items
        'kurczak': 'chicken',
        'wołowina': 'beef',
        'ser': 'cheese',
        'sałata': 'lettuce',
        'pomidor': 'tomato',
        'cebula': 'onion',
        
        # Size corrections
        'średnie': 'średni',
        'duze': 'duży',
        'male': 'mały',
    }
    
    # Add number patterns for Polish
    result['number_patterns'] = [
        'jeden', 'dwa', 'trzy', 'cztery', 'pięć', 'sześć', 'siedem', 'osiem', 'dziewięć', 'dziesięć',
        'jedna', 'dwie', 'jedną', 'dwoma', 'trzema',
    ]
    
    return result

if __name__ == "__main__":
    import os
    
    # Use the bk_menu.json from the root directory
    menu_file = os.path.join(os.path.dirname(__file__), '../../../bk_menu.json')
    menu_vocab = extract_bk_menu_vocabulary(menu_file)
    
    # Save to JSON
    output_file = os.path.join(os.path.dirname(__file__), 'bk_menu_vocabulary.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(menu_vocab, f, indent=2, ensure_ascii=False)
    
    # Save all unique items as text list
    items_file = os.path.join(os.path.dirname(__file__), 'bk_menu_items_all.txt')
    with open(items_file, 'w', encoding='utf-8') as f:
        for item in menu_vocab['all_items']:
            f.write(f"{item}\n")
    
    print(f"Extracted {len(menu_vocab['all_items'])} unique BK Poland menu items")
    print(f"Categories: {list(menu_vocab.keys())}")
    print(f"Saved to: {output_file}")