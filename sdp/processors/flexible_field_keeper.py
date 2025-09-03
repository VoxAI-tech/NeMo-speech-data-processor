"""Flexible field keeper that handles missing fields gracefully."""

import json
from typing import List
from tqdm import tqdm
from sdp.processors.base_processor import BaseProcessor


class FlexibleKeepFields(BaseProcessor):
    """
    Keep only specified fields, but handle missing fields gracefully.
    """
    
    def __init__(self, fields_to_keep: List[str], **kwargs):
        super().__init__(**kwargs)
        self.fields_to_keep = fields_to_keep
        self.missing_fields_count = {}
    
    def process(self):
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin, \
             open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            
            for line in tqdm(fin, desc="Keeping specified fields"):
                entry = json.loads(line)
                new_entry = {}
                
                for field in self.fields_to_keep:
                    if field in entry:
                        new_entry[field] = entry[field]
                    else:
                        # Track missing fields but don't fail
                        if field not in self.missing_fields_count:
                            self.missing_fields_count[field] = 0
                        self.missing_fields_count[field] += 1
                
                fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
        
        # Report missing fields
        if self.missing_fields_count:
            print("\n=== Missing Fields Report ===")
            for field, count in self.missing_fields_count.items():
                print(f"  Field '{field}' was missing in {count} entries")