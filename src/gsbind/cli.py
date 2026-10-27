import argparse
import json
import os
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description='GSbind E11 paper engineering and evidence')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/main.json')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['train', 'preprocess', 'infer', 'evaluate', 'select-checkpoint', 'show-config']:
        commands.add_parser(name)
    rebuild_pinder = commands.add_parser(
        'rebuild-pinder',
        help='Rebuild the GSBind training/test split from the public PINDER API',
    )
    rebuild_pinder.add_argument('--output-dir', type=Path, required=True)
    rebuild_pinder.add_argument('--release', default='2024-02')
    rebuild_pinder.add_argument('--cache-dir', type=Path)
    rebuild_pinder.add_argument('--workers', type=int, default=8)
    rebuild_pinder.add_argument('--download-workers', type=int, default=8)
    rebuild_pinder.add_argument('--chunk-size', type=int, default=512)
    rebuild_pinder.add_argument('--limit', type=int, default=0)
    rebuild_pinder.add_argument('--cluster-cap', type=int, default=5)
    rebuild_pinder.add_argument('--contact-cutoff', type=float, default=5.0)
    rebuild_pinder.add_argument('--anchor-manifest', type=Path)
    rebuild_pinder.add_argument(
        '--anchor-only',
        action='store_true',
        help='Reproduce only the holo/side targets in the anchor manifest from official API sources',
    )
    rebuild_pinder.add_argument('--update-index', action='store_true')
    validate_pinder = commands.add_parser('validate-pinder-splits')
    validate_pinder.add_argument('--cache-dir', type=Path)
    validate_pinder.add_argument('--train', type=Path)
    validate_pinder.add_argument('--holo-test', type=Path)
    validate_pinder.add_argument('--apo-af2-test', type=Path)
    validate_pinder.add_argument('--output', type=Path)
    evidence = commands.add_parser('evidence')
    evidence.add_argument('--task', choices=['all', 'main', 'ssm220', 'downstream'], default='all')
    summary = commands.add_parser('summarize', help='Recompute paper metrics from public saved predictions; no model fitting')
    summary.add_argument('--task', choices=['all', 'main', 'ssm220', 'downstream'], default='all')
    summary.add_argument('--output', type=Path, default=ROOT / 'runs/public_recompute')
    downstream = commands.add_parser('downstream', help='AlloBench source-selected heads, frozen AM transfer and Src selected OOF')
    downstream.add_argument('--action', choices=['replay', 'train'], default='replay')
    downstream.add_argument('--output', type=Path, default=ROOT / 'runs/downstream')
    downstream.add_argument('--task', choices=['all', 'AlloBench', 'AM', 'Src'], default='all')
    verification = commands.add_parser('verify')
    verification.add_argument('--hashes', action='store_true')
    commands.add_parser('figures')
    commands.add_parser('rebuild-ssm-labels')
    commands.add_parser('rebuild-allosteric-labels')
    commands.add_parser('rebuild-functional-labels')
    covariates = commands.add_parser('covariates')
    covariates.add_argument('--manifest', type=Path, required=True)
    covariates.add_argument('--output', type=Path, required=True)
    surface = commands.add_parser('surface-forward')
    surface.add_argument('--manifest', type=Path, required=True)
    surface.add_argument('--output', type=Path, required=True)
    surface.add_argument('--device', default='cpu')
    surface.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.config = args.config.resolve()
    if args.command in ['surface-forward', 'covariates']:
        args.manifest = args.manifest.resolve()
        args.output = args.output.resolve()
    elif args.command == 'rebuild-pinder':
        args.output_dir = args.output_dir.resolve()
        args.cache_dir = args.cache_dir.resolve() if args.cache_dir else None
        args.anchor_manifest = args.anchor_manifest.resolve() if args.anchor_manifest else None
    elif args.command == 'validate-pinder-splits':
        args.train = (args.train or ROOT / 'datasets/main/train.csv').resolve()
        args.holo_test = (args.holo_test or ROOT / 'datasets/main/holo_test.csv').resolve()
        args.apo_af2_test = (args.apo_af2_test or ROOT / 'datasets/main/apo_af2_test.csv').resolve()
        args.output = args.output.resolve() if args.output else None
    if args.command in ['downstream', 'summarize']:
        args.output = args.output.resolve()
    os.chdir(ROOT)
    if args.command == 'downstream':
        from .downstream import replay
        tasks = None if args.task == 'all' else [args.task]
        if args.action == 'train':
            from .downstream_training import train
            print(train(ROOT, args.output, tasks).to_string(index=False))
        else:
            print(replay(ROOT, args.output, tasks).to_string(index=False))
    elif args.command == 'summarize':
        from .evidence import run
        run(ROOT, args.task, output_root=args.output)
    elif args.command == 'evidence':
        from .evidence import run
        run(ROOT, args.task)
    elif args.command == 'verify':
        from .verification import verify, verify_hashes
        print(json.dumps(verify_hashes(ROOT) if args.hashes else verify(ROOT), indent=2))
    elif args.command == 'figures':
        from .figures import build
        build(ROOT)
    elif args.command == 'rebuild-ssm-labels':
        from .labels import rebuild_ssm
        rebuild_ssm(ROOT, ROOT / 'runs/ssm_labels')
    elif args.command == 'rebuild-allosteric-labels':
        from .labels import rebuild_allosteric
        rebuild_allosteric(ROOT, ROOT / 'runs/allosteric_labels')
    elif args.command == 'rebuild-functional-labels':
        from .labels import rebuild_functional
        rebuild_functional(ROOT, ROOT / 'runs/functional_labels')
    elif args.command == 'covariates':
        from .covariates import rebuild
        rebuild(ROOT, args.manifest, args.output)
    elif args.command == 'surface-forward':
        from .forward import run
        run(ROOT, args.manifest, args.output, args.device, args.limit)
    elif args.command == 'rebuild-pinder':
        from .pinder_pipeline import main as rebuild_pinder
        report = rebuild_pinder([
            '--output-dir', str(args.output_dir),
            '--release', args.release,
            *([] if args.cache_dir is None else ['--cache-dir', str(args.cache_dir)]),
            '--workers', str(args.workers),
            '--download-workers', str(args.download_workers),
            '--chunk-size', str(args.chunk_size),
            '--limit', str(args.limit),
            '--cluster-cap', str(args.cluster_cap),
            '--contact-cutoff', str(args.contact_cutoff),
            *([] if args.anchor_manifest is None else ['--anchor-manifest', str(args.anchor_manifest)]),
            *(['--anchor-only'] if args.anchor_only else []),
            *(['--update-index'] if args.update_index else []),
        ])
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    elif args.command == 'validate-pinder-splits':
        from .pinder_pipeline import validate_published_pinder_splits_main
        report = validate_published_pinder_splits_main([
            *([] if args.cache_dir is None else ['--cache-dir', str(args.cache_dir)]),
            '--train', str(args.train),
            '--holo-test', str(args.holo_test),
            '--apo-af2-test', str(args.apo_af2_test),
            *([] if args.output is None else ['--output', str(args.output)]),
        ])
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        from gaussian_direct.app import GaussianDirectApp
        from gaussian_direct.config import ExperimentConfig
        config = ExperimentConfig.from_file(args.config)
        if args.command == 'show-config':
            print(json.dumps(config.to_dict(), indent=2))
        else:
            getattr(GaussianDirectApp(config), args.command)()
