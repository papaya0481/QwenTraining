# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, Optional

from swift.dataset import (
    DatasetMeta, load_dataset, register_dataset, MessagesPreprocessor,
    SubsetDataset
)


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
        dataset_name='codegen1_train',
        preprocess_func=CodeGen1SFTPreprocessor(),
        subsets=[SubsetDataset('sft', subset='sft', split=['train']),
                 SubsetDataset('rl', subset='rl', split=['train']),
                 SubsetDataset('defaults', subset='defaults', split=['train']),
                 ],
    )
)

register_dataset(
    DatasetMeta(
        hf_dataset_id='BigfufuOuO/codegen1_merged_clean',
        dataset_name='codegen1_sft_val',
        preprocess_func=CodeGen1SFTPreprocessor(),
        subsets=[SubsetDataset('sft', subset='sft', split=['test']),]
    )
)

if __name__ == '__main__':
    dataset = load_dataset('codegen1_train:sft', use_hf=True)[0]
    print(f'dataset: {dataset}')
    
    dataset = load_dataset('codegen1_train:rl', use_hf=True)[0]
    print(f'dataset: {dataset}')

    dataset = load_dataset('codegen1_sft_val', use_hf=True)[0]
    print(f'dataset: {dataset}')