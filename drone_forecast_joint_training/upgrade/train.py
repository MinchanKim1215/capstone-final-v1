"""학습/검증만 이용한 Transformer 개선 실험. 시험셋은 이 모듈에서 열지 않습니다."""
from __future__ import annotations

from pathlib import Path
import argparse
import copy
import json
import math
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from upgrade.data import Bank, PREPARED
from upgrade.model import LinkTransformer, contiguous_hidden, parameter_count
from upgrade.map_setup import digest

RUNS = ROOT / 'results' / 'upgrade' / 'runs'
VARIANTS = {
    'A0_three_regions': {'bank': 'train3', 'normalizer': 'train3', 'features': 12, 'pretrain': False, 'extra_updates': 0},
    'A1_more_maps': {'bank': 'train30', 'normalizer': 'train30', 'features': 12, 'pretrain': False, 'extra_updates': 0},
    'A2_observed_geometry': {'bank': 'train30', 'normalizer': 'train30', 'features': 28, 'pretrain': False, 'extra_updates': 0},
    'A3_masked_pretraining': {'bank': 'train30', 'normalizer': 'train30', 'features': 28, 'pretrain': True, 'extra_updates': 0},
    'A4_matched_compute': {'bank': 'train30', 'normalizer': 'train30', 'features': 28, 'pretrain': False, 'extra_updates': 500},
}
DEFAULT = {'updates': 2000, 'pretrain_updates': 500, 'batch_size': 256, 'learning_rate': .0003,
           'weight_decay': .0001, 'gradient_clip': 1., 'eval_every': 200, 'early_stop_checks': 4,
           'minimum_updates_before_stop': 800, 'validation_batch_size': 512, 'threads': 2,
           'first_seed': 11, 'replication_seeds': [22, 33], 'fpr_budget': .05,
           'checkpoint_selection': 'highest unweighted mean AP over seven validation regions',
           'variant_selection': 'highest validation macro AP, never test; A4 matches A3 maximum total updates',
           'model_family': 'one lightweight multivariate Transformer; no ensemble'}


def configure(seed, threads=2):
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # CPU 결정성: 같은 환경/버전/시드 재현을 위한 설정입니다.
    torch.use_deterministic_algorithms(True)


def macro_ap(y, p, region):
    by = {}
    for key in sorted(set(region.tolist())):
        take = region == key
        if 0 < y[take].sum() < take.sum():
            by[key] = float(average_precision_score(y[take], p[take]))
    return float(np.mean(list(by.values()))), by


@torch.inference_mode()
def predict_bank(model, bank, batch_size=512):
    model.eval()
    out = np.empty(len(bank), np.float32)
    for first in range(0, len(bank), batch_size):
        idx = np.arange(first, min(first+batch_size, len(bank)))
        x, m, _ = bank.batch(idx)
        out[idx] = torch.sigmoid(model(x, m)).numpy()
    return out


class Batches:
    def __init__(self, bank, batch_size, rng):
        self.bank, self.batch_size, self.rng = bank, batch_size, rng
        self.order = np.empty(0, int); self.position = 0

    def next(self):
        if self.position+self.batch_size > len(self.order):
            self.order = self.rng.permutation(len(self.bank)); self.position = 0
        idx = self.order[self.position:self.position+self.batch_size]
        self.position += len(idx)
        return self.bank.batch(idx)


def lr_at(step, total, start):
    warm = max(1, int(total*.05))
    if step < warm:
        return start*(step+1)/warm
    progress = (step-warm)/max(1, total-warm)
    return start*(.1+.9*(1+math.cos(math.pi*progress))/2)


def freeze_training_plan(overrides=None):
    path = ROOT / 'configs' / 'upgrade_training.json'
    config = DEFAULT | (overrides or {})
    if path.exists():
        previous = read_json(path)
        if previous['training'] != config or previous['variants'] != VARIANTS:
            raise ValueError('동결한 학습 계획과 다릅니다. 결과를 보고 계획을 덮어쓰지 마십시오.')
        return previous
    protocol = read_json(ROOT / 'configs' / 'upgrade_protocol.json')
    plan = {'protocol_sha256': protocol['protocol_sha256'], 'training': config, 'variants': VARIANTS,
            'model_spec': protocol['model_start_spec'], 'before_test_predictions': True,
            'code_sha256': {n: sha256_file(ROOT/n) for n in ('upgrade/model.py', 'upgrade/data.py', 'upgrade/train.py')}}
    plan['plan_sha256'] = digest(plan)
    write_json(path, plan)
    return plan


def train_one(variant, seed=11, plan=None):
    plan = plan or freeze_training_plan()
    config, spec = plan['training'], plan['variants'][variant]
    folder = RUNS / f'{variant}_s{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    done = folder/'run.json'
    if done.exists():
        old = read_json(done)
        if old['plan_sha256'] != plan['plan_sha256']:
            raise ValueError('이전 실험과 학습 계획이 다릅니다.')
        print(json.dumps({'stage': 'training_reused', 'run': folder.name}), flush=True)
        return old
    configure(seed, config['threads'])
    normalizer = read_json(PREPARED/f'normalizer_{spec["normalizer"]}.json')
    train = Bank(spec['bank'], normalizer, spec['features'])
    val = Bank('val7', normalizer, spec['features'])
    model = LinkTransformer(features=spec['features'], **plan['model_spec'])
    started = time.perf_counter()
    history, prehistory = [], []
    if spec['pretrain']:
        pretrain = Bank('pretrain30', normalizer, spec['features'])
        rng = np.random.default_rng(seed+771)
        stream = Batches(pretrain, config['batch_size'], rng)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        model.train()
        for step in range(config['pretrain_updates']):
            x, m, _ = stream.next()
            hidden = contiguous_hidden(m, rng)
            for group in optimizer.param_groups:
                group['lr'] = lr_at(step, config['pretrain_updates'], config['learning_rate'])
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(x, m, hidden, reconstruct=True)
            loss = ((reconstruction[hidden]-x[hidden])**2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('사전학습 손실이 유한하지 않습니다.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip']); optimizer.step()
            if step == 0 or (step+1) % 100 == 0:
                row = {'step': step+1, 'masked_mse': float(loss.detach()), 'elapsed_s': time.perf_counter()-started}
                prehistory.append(row)
                write_json(folder/'pretraining_history.json', prehistory)
                print(json.dumps({'stage': 'pretrain', 'run': folder.name, **row}), flush=True)
        torch.save({'state_dict': model.state_dict(), 'spec': model.spec, 'normalizer': normalizer}, folder/'pretrained.pt')
    # 복원 학습에 사용한 optimizer를 버리고 예측 손실에 대해 새로 미세조정합니다.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(seed+1401)
    stream = Batches(train, config['batch_size'], rng)
    maximum = config['updates']+spec['extra_updates']
    best, no_gain, best_step, cumulative_loss = -1., 0, 0, 0.
    for step in range(maximum):
        model.train()
        x, m, y = stream.next()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, maximum, config['learning_rate'])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, m), y)
        if not torch.isfinite(loss):
            raise RuntimeError('지도학습 손실이 유한하지 않습니다.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip']); optimizer.step()
        cumulative_loss += float(loss.detach())
        if (step+1) % config['eval_every'] == 0 or step+1 == maximum:
            p = predict_bank(model, val, config['validation_batch_size'])
            score, by = macro_ap(val.selected[:, 4], p, val.region)
            row = {'step': step+1, 'approximate_train_passes': (step+1)*config['batch_size']/len(train),
                   'training_bce': cumulative_loss/config['eval_every'], 'validation_macro_ap': score,
                   'validation_ap_by_region': by, 'elapsed_s': time.perf_counter()-started}
            history.append(row); cumulative_loss = 0.
            if score > best+1e-6:
                best, no_gain, best_step = score, 0, step+1
                torch.save({'state_dict': model.state_dict(), 'spec': model.spec, 'normalizer': normalizer,
                            'variant': variant, 'seed': seed, 'step': best_step, 'plan_sha256': plan['plan_sha256']}, folder/'best.pt')
                np.save(folder/'validation_bank_probability.npy', p)
            else:
                no_gain += 1
            write_json(folder/'history.json', history)
            print(json.dumps({'stage': 'train', 'run': folder.name, 'step': step+1,
                              'val_macro_ap': round(score, 6), 'best_step': best_step,
                              'elapsed_s': round(time.perf_counter()-started, 1)}), flush=True)
            if step+1 >= config['minimum_updates_before_stop'] and no_gain >= config['early_stop_checks']:
                break
    result = {'variant': variant, 'seed': seed, 'complete': True, 'plan_sha256': plan['plan_sha256'],
              'model_spec': model.spec, 'parameters': parameter_count(model), 'training_regions': train.meta['regions'],
              'training_samples_per_pass': len(train), 'validation_samples': len(val), 'pretraining_updates': config['pretrain_updates'] if spec['pretrain'] else 0,
              'supervised_updates': step+1, 'best_step': best_step, 'validation_macro_ap': best,
              'elapsed_s': time.perf_counter()-started, 'test_used': False,
              'best_checkpoint_sha256': sha256_file(folder/'best.pt'),
              'code_sha256': {n: sha256_file(ROOT/n) for n in ('upgrade/model.py', 'upgrade/data.py', 'upgrade/train.py')}}
    write_json(done, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANTS)+['all'], default='all')
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()
    plan = freeze_training_plan()
    for variant in VARIANTS if args.variant == 'all' else [args.variant]:
        train_one(variant, args.seed, plan)


if __name__ == '__main__':
    main()
