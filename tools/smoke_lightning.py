import json
import os
from pathlib import Path
import sys

import torch
from lightning.pytorch import Trainer, seed_everything

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gaussian_direct.config import ExperimentConfig
from gaussian_direct.lightning import GaussianDirectDataModule, GaussianDirectLightningModule


def main():
    os.chdir(ROOT)
    torch.set_num_threads(2)
    seed_everything(42, workers=True)
    config = ExperimentConfig.from_file(ROOT / 'configs/smoke.json')
    module = GaussianDirectLightningModule(config, steps_per_epoch=2, pos_weight=torch.tensor(1.))
    datamodule = GaussianDirectDataModule(config)
    trainer = Trainer(accelerator='cpu', devices=1, precision='32-true', fast_dev_run=1,
                      logger=False, enable_checkpointing=False, enable_model_summary=False)
    trainer.fit(module, datamodule=datamodule)
    result = dict(status='passed', optimizer_steps=trainer.global_step, main_checkpoint_modified=False,
                  purpose='One CPU training/gradient/optimizer step; not reproduction of twenty-epoch training')
    (ROOT / 'provenance/lightning_smoke.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
