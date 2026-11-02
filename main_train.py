"""Five-fold PCMCI, raster training, and 2017 evaluation."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

# Locate projection data in the active environment.
_rasterio_data = Path(sys.prefix)/'Lib'/'site-packages'/'rasterio'
_library = Path(sys.base_prefix)/'Library'/'share'
_proj = _rasterio_data/'proj_data' if (_rasterio_data/'proj_data'/'proj.db').exists() else _library/'proj'
_gdal = _rasterio_data/'gdal_data' if (_rasterio_data/'gdal_data').exists() else _library/'gdal'
if (_proj/'proj.db').exists():
    os.environ['PROJ_LIB'] = str(_proj)
    os.environ['PROJ_DATA'] = str(_proj)
    os.environ['GDAL_DATA'] = str(_gdal)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_loader import RasterStore, GridDataset, INPUTS, fit_stats
from cig_sftnet import CIG_SFTNet
from experiment import (development_dates, prepare, write_json, masked_l1,
                        read_stations, locate_stations, paired_metrics)
from pcmci import fit_pcmci, build_station_seeds
from inference import tiled_predict, save_geotiff, load_checkpoint_model

HERE = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    out = Path(cfg['output_dir'])
    cfg['output_dir'] = str(out if out.is_absolute() else HERE/out)
    return cfg


def pcmci_runtime_options(cfg):
    prepared = Path(cfg['output_dir'])/'prepared'
    folds = json.loads((prepared/'folds.json').read_text(encoding='utf-8'))
    seeds = build_station_seeds(folds, cfg['seed'])
    record = {'policy': 'first_fold_train_then_test_v1', 'base_seed': cfg['seed'],
              'station_seeds': seeds}
    path = prepared/'pcmci_station_seeds.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Fixed PCMCI station seeds changed: use a new output_dir')
    else:
        write_json(path, record)
    return dict(cfg['pcmci'], seed=cfg['seed'], station_seeds=seeds,
                cache_dir=Path(cfg['output_dir'])/'pcmci_cache')


def checkpoint_payload(model, causal_path, stats, fold, **extra):
    return dict(model_state=model.state_dict(), causal_json=str(Path(causal_path).resolve()),
                causal_structure=model.causal_structure, stats=stats, fold=fold,
                model_sha256=hashlib.sha256((HERE/'cig_sftnet.py').read_bytes()).hexdigest(), **extra)


def save_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); temporary.replace(path)


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    model.train(optimizer is not None)
    absolute_error, count = 0., 0.
    for batch in loader:
        inputs = {k: batch[k].to(device) for k in INPUTS}
        target = batch['Label'].to(device)
        mask = batch['BoundaryMask'].to(device)
        n = mask.sum().item()
        if n == 0:
            continue
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                pred = model(inputs)
                loss = masked_l1(pred, target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite grid loss')
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update()
        absolute_error += loss.item()*n
        count += n
    if count == 0:
        raise ValueError('No valid grid pixels in epoch')
    return absolute_error/count


def train_one(store, train_dates, val_dates, stats, causal, folder, fold, cfg, lr, epochs, device):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    seed_all(cfg['seed']+fold)
    model = CIG_SFTNet(str(causal)).to(device)
    # Cache whole grids separately for training and validation.
    def preload(dates):
        dataset = GridDataset(store, dates, stats)
        samples = []
        for index in range(len(dataset)):
            samples.append(dataset[index])
            if index % 100 == 0:
                print('Fold %d preload %d/%d grids' % (fold,index+1,len(dataset)), flush=True)
        return samples

    loader = DataLoader(preload(train_dates), batch_size=cfg['batch_size'],
                        shuffle=True, num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = (DataLoader(preload(val_dates), batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=0) if val_dates else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best, best_epoch, history = float('inf'), 0, []
    scale = stats['Label'][1]-stats['Label'][0]
    scale = scale if scale > 0 else 1.
    for epoch in range(1, epochs+1):
        start = time.monotonic()
        current_lr = optimizer.param_groups[0]['lr']
        # Refit uses the learning-rate schedule selected on 2016.
        if not val_dates and 'refit_lr_schedule' in cfg:
            current_lr = cfg['refit_lr_schedule'][epoch-1]
            for group in optimizer.param_groups:
                group['lr'] = current_lr
        loss = run_epoch(model, loader, device, optimizer, scaler)
        val = run_epoch(model, val_loader, device) if val_loader else None
        history.append({'epoch': epoch, 'lr': current_lr, 'train_mae_c': loss*scale,
                        'val_mae_c': val*scale if val is not None else None,
                        'seconds': time.monotonic()-start})
        write_json(folder/'history.json', history)
        print('Fold %d epoch %d/%d train %.4f C val %s (%.1fs)' %
              (fold, epoch, epochs, loss*scale, str(val*scale if val is not None else None), history[-1]['seconds']), flush=True)
        if val is not None:
            scheduler.step(val)
            if val < best:
                best, best_epoch = val, epoch
                save_checkpoint(folder/'best.pt', checkpoint_payload(model, causal, stats, fold, epoch=epoch))
            if epoch-best_epoch >= cfg['early_stopping_patience']:
                break
        else:
            save_checkpoint(folder/'last.pt', checkpoint_payload(model, causal, stats, fold, epoch=epoch))
    if val_loader is None:
        return model, history
    return {'lr': lr, 'best_epoch': best_epoch, 'val_mae_c': best*scale,
            'lr_schedule': [r['lr'] for r in history[:best_epoch]], 'history': str(folder/'history.json')}


def train_fold(cfg, fold, device):
    out = Path(cfg['output_dir']); folder = out/('fold_%d' % fold['fold'])
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'final.pt').exists():
        raise FileExistsError(str(folder/'final.pt') + ' already exists; choose a new output_dir for a new experiment')
    store = RasterStore(cfg['data_root'])
    table = pd.read_csv(out/'prepared'/'station_features_2015_2016.csv', dtype={'station_id': str}, parse_dates=['date'])
    pcmci_options = pcmci_runtime_options(cfg)
    train, val, refit = development_dates()
    write_json(folder/'protocol.json', dict(config=cfg, fold=fold,
                inner_train=['2015-01-01','2015-12-31'], inner_val=['2016-01-01','2016-12-31'],
                final_refit=['2015-01-01','2016-12-31'], independent_test=['2017-01-01','2017-12-31']))
    print('Fold %d: fit 2015 normalization and PCMCI' % fold['fold'], flush=True)
    stats = fit_stats(store, train)
    write_json(folder/'inner_stats.json', stats)
    causal = fit_pcmci(table, fold['train_stations'], train, folder/'inner_pcmci',
                       **pcmci_options)
    trials = []
    for i, lr in enumerate(cfg['learning_rates']):
        trials.append(train_one(store, train, val, stats, causal, folder/('trial_%d' % (i+1)),
                                fold['fold'], cfg, lr, cfg['epochs'], device))
    selected = min(trials, key=lambda row: row['val_mae_c'])
    write_json(folder/'selection.json', {'trials': trials, 'selected': selected,
                                        'selection_source': '2016 T2m raster MAE only'})
    print('Fold %d: refit preprocessing on 2015-2016' % fold['fold'], flush=True)
    final_stats = fit_stats(store, refit)
    write_json(folder/'final_stats.json', final_stats)
    final_causal = fit_pcmci(table, fold['train_stations'], refit, folder/'final_pcmci',
                             **pcmci_options)
    refit_cfg = dict(cfg, refit_lr_schedule=selected['lr_schedule'])
    model, _ = train_one(store, refit, [], final_stats, final_causal, folder/'refit', fold['fold'],
                         refit_cfg, selected['lr'], selected['best_epoch'], device)
    save_checkpoint(folder/'final.pt', checkpoint_payload(model, final_causal, final_stats, fold['fold'],
                    selected=selected, train_station_ids=fold['train_stations'], test_station_ids=fold['test_stations'],
                    train_years=[2015,2016], independent_test_year=2017))


def predict_fold(cfg, fold, device, start='2017-01-01', end='2017-12-31'):
    dates = list(pd.date_range(start, end))
    if not dates or any(d.year != 2017 for d in dates):
        raise ValueError('Independent test dates must be in 2017')
    folder = Path(cfg['output_dir'])/('fold_%d' % fold['fold'])
    model, checkpoint = load_checkpoint_model(folder/'final.pt', device)
    if checkpoint['test_station_ids'] != fold['test_stations']:
        raise ValueError('Checkpoint station fold mismatch')
    fine, coarse = RasterStore(cfg['data_root'], 'fine'), RasterStore(cfg['data_root'])
    obs = read_stations(cfg['data_root'], [2017])
    obs = obs.loc[obs.station_id.isin(fold['test_stations']) & obs.date.isin(dates)].copy()
    fine_obs, coarse_obs = locate_stations(obs, fine), locate_stations(obs, coarse)
    rows = []
    full_year = start == '2017-01-01' and end == '2017-12-31'
    target = folder/('test_2017' if full_year else 'test_%s_%s' % (start.replace('-',''), end.replace('-','')))
    target.mkdir(parents=True, exist_ok=True)
    for date in dates:
        prediction = tiled_predict(model, fine, date, checkpoint['stats'], device,
                                    tile_size=cfg['tile_size'], halo=cfg['halo'])
        save_geotiff(target/'tif'/('PGC_SFTNet_%s.tif' % date.strftime('%Y%m%d')), prediction, fine.profile)
        baseline = coarse.read('Label', date)
        for index in obs.index[obs.date.eq(date)]:
            fr, fc = fine_obs.loc[index, ['row','col']].astype(int)
            cr, cc = coarse_obs.loc[index, ['row','col']].astype(int)
            after = prediction[fr,fc] if 0<=fr<fine.shape[0] and 0<=fc<fine.shape[1] else np.nan
            before = baseline[cr,cc] if 0<=cr<coarse.shape[0] and 0<=cc<coarse.shape[1] else np.nan
            rows.append({'station_id': obs.loc[index,'station_id'], 'date': str(date.date()),
                         'observed_c': obs.loc[index,'TEMP'], 'before_c': before, 'after_c': after,
                         'fold': fold['fold']})
        print('Fold %d downscaled %s' % (fold['fold'], date.date()), flush=True)
    result = pd.DataFrame(rows, columns=['station_id','date','observed_c','before_c','after_c','fold'])
    result.to_csv(target/'station_pairs.csv', index=False)
    metric = paired_metrics(result.observed_c, result.before_c, result.after_c)
    metric.update({'start': start, 'end': end, 'requested_stations': fold['test_stations'],
                   'n_observation_rows': len(result), 'paired_rows_excluded': len(result)-metric['n']})
    write_json(target/'metrics.json', metric)
    by_station = {sid: paired_metrics(g.observed_c,g.before_c,g.after_c) for sid,g in result.groupby('station_id')}
    write_json(target/'station_metrics.json', by_station)
    if metric['n'] > 10:
        from plot_scatter import plot_pairs
        plot_pairs(target/'station_pairs.csv', target/'scatter')
    else:
        print('Skip scatter plot: fewer than 11 shared valid pairs.', flush=True)


def summarize(cfg):
    files = sorted(Path(cfg['output_dir']).glob('fold_*/test_2017/station_pairs.csv'))
    if not files:
        return
    rows = pd.concat([pd.read_csv(p, dtype={'station_id':str}) for p in files], ignore_index=True)
    if rows.duplicated(['station_id','date']).any():
        raise ValueError('Duplicate held-out station/date across folds')
    rows.to_csv(Path(cfg['output_dir'])/'all_test_pairs.csv', index=False)
    metrics = dict(paired_metrics(rows.observed_c,rows.before_c,rows.after_c), completed_folds=len(files))
    write_json(Path(cfg['output_dir'])/'all_test_metrics.json', metrics)
    if metrics['n'] > 10:
        from plot_scatter import plot_pairs
        plot_pairs(Path(cfg['output_dir'])/'all_test_pairs.csv', Path(cfg['output_dir'])/'scatter_all_folds')
    else:
        print('Skip scatter plot: fewer than 11 shared valid pairs.', flush=True)


@contextmanager
def cli_run_guard(cfg):
    """Lock output writes and verify any existing experiment manifest."""
    if (HERE/'.project_maintenance.json').exists():
        raise RuntimeError('Project maintenance is in progress; do not start another run')
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    lock = out/'formal_run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        manifest_path = out/'formal_manifest.json'
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding='utf-8'))
            sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
            if previous['config'] != cfg or previous['source_sha256'] != sources:
                raise ValueError('Experiment configuration/source changed: use a new output_dir')
            for name, expected in previous['prepared_sha256'].items():
                if hashlib.sha256((out/'prepared'/name).read_bytes()).hexdigest() != expected:
                    raise ValueError('Experiment prepared data changed: use a new output_dir')
        yield
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','train','predict','all'])
    parser.add_argument('--config', default=str(HERE/'config.json'))
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1,6), default=list(range(1,6)))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--start', default='2017-01-01')
    parser.add_argument('--end', default='2017-12-31')
    args = parser.parse_args(); cfg = load_config(args.config)
    with cli_run_guard(cfg):
        device = torch.device(args.device)
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        print('Device:', device, 'Output:', cfg['output_dir'], flush=True)
        if args.stage in ('prepare','all'):
            prepare(cfg['data_root'], Path(cfg['output_dir'])/'prepared', cfg['seed'])
        if args.stage == 'prepare':
            return
        folds = json.loads((Path(cfg['output_dir'])/'prepared'/'folds.json').read_text(encoding='utf-8'))
        for fold in folds:
            if fold['fold'] not in args.folds:
                continue
            if args.stage in ('train','all'):
                train_fold(cfg, fold, device)
            if args.stage in ('predict','all'):
                predict_fold(cfg, fold, device, args.start, args.end)
        if args.stage in ('predict','all'):
            summarize(cfg)


if __name__ == '__main__':
    main()
