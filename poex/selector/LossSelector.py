from .selector import SelectPolicy
from ..datasets import JailbreakDataset
from ..utils import model_utils
from ..models import WhiteBoxModelBase

import warnings
import torch
import logging
from tqdm import tqdm

class LossSelector(SelectPolicy):
    def __init__(self, model:WhiteBoxModelBase, evaluator:None, batch_size=None, is_universal=False, alpha=10):
        assert isinstance(model, WhiteBoxModelBase)
        self.model = model
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.is_universal = is_universal
        self.alpha = alpha  # Weight for balancing reference loss and policy score

    def select(self, dataset)->JailbreakDataset:
        if not self.is_universal and len(dataset.group_by_parents()) > 1:
            return JailbreakDataset.merge([self.select(JailbreakDataset(group)) for group in tqdm(dataset.group_by_parents(), desc='Reference Loss Score Selection')])

        if self.batch_size is None:
            batches = [dataset]
        else:
            batches = [dataset[i: i+self.batch_size] for i in range(0, len(dataset), self.batch_size)]

        with torch.no_grad():
            for batch in batches:
                B = len(batch)
                
                # Calculate reference loss
                batch_input_ids = []
                batch_labels = []
                for instance in batch:
                    assert len(instance.reference_responses) >= 1
                    if len(instance.reference_responses) > 1:
                        warnings.warn(f'传入`ReferenceLossSelector`的每个instance的reference_responses大小都为1，而不是{len(instance.reference_responses)}。将默认使用第一个。')

                    input_ids, _, _, target_slice = model_utils.encode_trace(self.model, instance.query, instance.jailbreak_prompt, instance.reference_responses[0])
                    labels = torch.full_like(input_ids, -100)
                    labels[:, target_slice] = input_ids[:, target_slice]
                    batch_input_ids.append(input_ids)
                    batch_labels.append(labels)
                batch_input_ids = model_utils.pad_and_stack(batch_input_ids, self.model.pad_token_id)
                batch_labels = model_utils.pad_and_stack(batch_labels, -100)

                batch_loss = model_utils.batch_loss(self.model, batch_input_ids, batch_labels)
                
                # Calculate policy score
                for idx, instance in enumerate(batch):
                    self.evaluator._evaluate(instance)
                    policy_score = int(instance.eval_policy_score[-1])

                    # Calculate all loss
                    all_loss = batch_loss[idx].item() + self.alpha * (4 - policy_score)
                    instance.all_loss = all_loss

        best_group = None
        best_loss = None
        for group in dataset.group_by(lambda x: x.jailbreak_prompt):
            total_loss = sum([instance.all_loss for instance in group])
            if best_loss is None or total_loss < best_loss:
                best_loss = total_loss
                best_group = group
        
        logging.info(f'best loss = {best_loss}')
        logging.info(f'best jailbreak prompt = `{best_group[0].jailbreak_prompt}`')

        return JailbreakDataset(best_group)
