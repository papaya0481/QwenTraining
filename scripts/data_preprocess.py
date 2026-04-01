# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, Optional

import swift
from swift.dataset import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset, MessagesPreprocessor


class CodeGen1SFTPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        system = row.get('system', '')
        user = row.get('user', '')
        assistant = row.get('assistant', '')
        row['messages'] = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': assistant},
        ]
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        hf_dataset_id='BigfufuOuO/codegen1_merged_clean',
        preprocess_func=CodeGen1SFTPreprocessor(),
    )
)

if __name__ == '__main__':
    dataset = load_dataset(['BigfufuOuO/codegen1_merged_clean'], use_hf=True)[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')